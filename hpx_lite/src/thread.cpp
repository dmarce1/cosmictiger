/*
 * thread.cpp
 *
 *  Created on: Nov 21, 2015
 *      Author: dmarce1
 */

#include <mpi.h>
#include <hwloc.h>

#include "hpx_lite/thread.hpp"

#include <cmath>
#include <cassert>
#include <ucontext.h>

#include <list>
#include <mutex>
#include <stack>
#include <thread>
#include <unordered_map>
#include <vector>
#include <queue>

#define HWLOC_OBJ_QUEUE HWLOC_OBJ_SOCKET

namespace hpx {

namespace detail {

constexpr std::size_t stack_size = 1024 * 1024;
constexpr std::size_t max_queue = 256;

class worker;

static void* stacks_alloc();
static void stacks_free(void*);
static std::shared_ptr<context> queue_pop(int);
static void queue_push(int, const std::shared_ptr<context>& ptr);
static void delayed_push(int);
static std::size_t& num_queues();

thread_exit::thread_exit() :
		exit_func(nullptr) {
}

thread_exit::thread_exit(const std::function<void()>& func) :
		exit_func(func) {
}

thread_exit::thread_exit(std::function<void()>&& func) :
		exit_func(std::move(func)) {
}

thread_exit& thread_exit::operator=(const std::function<void()>& func) {
	exit_func = func;
	return *this;
}

thread_exit& thread_exit::operator=(std::function<void()>&& func) {
	exit_func = std::move(func);
	return *this;
}

thread_exit::~thread_exit() {
	if (exit_func != nullptr) {
		exit_func();
	}
}

static bool& ready() {
	static bool a(false);
	return a;
}

static hwloc_topology_t& topology() {
	static hwloc_topology_t a;
	return a;
}

static bool& finalize() {
	static bool a(false);
	return a;
}

class context: public ucontext_t {
private:
	bool active; //
	bool stack_del;
	std::list<hpx::detail::thread_exit> exit_funcs;
	std::queue<std::function<void()>> continuations;
	bool continuations_closed;
	mutable std::mutex mtx;
	std::shared_ptr<bool> yield_bit;
public:
	context(const context&) = delete;
	context& operator=(const context&) = delete;
	context(context&&) = delete;
	context& operator=(context&&) = delete;
	void add_exit_function(std::function<void()>&&);
	bool is_active() const; //
	static void wrapper(void* func_ptr, void* arg_ptr, void* ctx_ptr);
	context();
	context(void (*func_ptr)(void*), void* arg_ptr);
	~context();
	bool add_continuation(std::function<void()>& func);
	bool get_yield_bit() const {
		bool rc;
		if (yield_bit == nullptr) {
			rc = true;
		} else {
			rc = *yield_bit;
		}
		return rc;
	}
	void set_yield_bit(const std::shared_ptr<bool>& b) {
		yield_bit = b;
	}
};


bool context::add_continuation(std::function<void()>& func) {
	std::unique_lock < std::mutex > lock(mtx);
	bool rc;
	if (!continuations_closed) {
		continuations.push(std::move(func));
		rc = true;
	} else {
		rc = false;
	}
	return rc;
}

class worker {
private:
	std::thread thread;
	std::shared_ptr<context> current_ctx;
	std::shared_ptr<context> main_ctx;
	std::shared_ptr<context> to_q_ctx;
	std::size_t proc_num;
	hwloc_cpuset_t cpuset;
public:
	static void main_context(std::reference_wrapper<worker>&& this_worker);
	worker(const std::size_t proc_num, const hwloc_obj_t&);
	void join();
	std::size_t queue_num() const {
		static const std::size_t nproc = std::thread::hardware_concurrency();
		return proc_num * num_queues() / nproc;
	}
	friend void hpx::this_thread::yield(const std::shared_ptr<bool>);
	friend void delayed_push(int);
	friend std::shared_ptr<context> thread_fork(void (*)(void*), void*);
	friend hpx::thread::id hpx::this_thread::get_id();
	friend void hpx::this_thread::detail::set_exit_function(std::function<void()>&&);

};

typedef std::unordered_map<pthread_t, std::reference_wrapper<worker>> worker_map_type;

static std::list<std::shared_ptr<context>>& queue(int i) {
	static std::vector<std::list<std::shared_ptr<context>>>a(max_queue);
	return a[i];
}

static worker_map_type& worker_map() {
	static worker_map_type a;
	return a;
}

static std::size_t& num_queues() {
	static std::size_t a;
	return a;
}

static std::mutex& worker_map_mutex() {
	static std::mutex a;
	return a;
}

static std::mutex& queue_mutex(int i) {
	static std::vector<std::mutex> a(max_queue);
	return a[i];
}

static std::vector<std::shared_ptr<worker>>& worker_ptr() {
	static std::vector<std::shared_ptr<worker>> a;
	return a;
}

static std::atomic<std::size_t>& worker_count() {
	static std::atomic<std::size_t> a(0);
	return a;
}

static std::stack<void*>& stacks_avail() {
	static std::stack<void*> a;
	return a;
}

static std::mutex& stacks_mutex() {
	static std::mutex a;
	return a;
}

static std::atomic<int>& nstacks() {
	static std::atomic<int> a(0);
	return a;
}

static std::atomic<std::size_t>& q_thread_cnt(int i) {
	if (i > 10000) {
		printf("Interal error!\n");
	}
	static std::atomic<std::size_t> a[10000];
	return a[i];
}

/*
 static void print_queues() {
 for (std::size_t i = 0; i != num_queues(); ++i) {
 printf("%5i ", int(q_thread_cnt(i)));
 }
 printf("\n");
 }*/

bool context::is_active() const {
	return active;
}

void context::add_exit_function(std::function<void()>&& f) {
	exit_funcs.push_back(std::move(f));
}

void context::wrapper(void* func_ptr, void* arg_ptr, void* _ctx_ptr) {
	context* ctx_ptr = reinterpret_cast<context*>(_ctx_ptr);
	int queue_num = 0;
	delayed_push(queue_num);
	(*(reinterpret_cast<void (*)(void*)>(func_ptr)))(arg_ptr);
	{
		std::unique_lock < std::mutex > lock(ctx_ptr->mtx);
		while (!ctx_ptr->continuations.empty()) {
			auto func = ctx_ptr->continuations.front();
			ctx_ptr->continuations.pop();
			lock.unlock();
			func();
			lock.lock();
		}
		ctx_ptr->continuations_closed = true;
	}
	ctx_ptr->active = false;
	hpx::this_thread::yield();
}

context::context() :
		active(true), stack_del(false), continuations_closed(false), yield_bit(nullptr) {
	if (getcontext(this) != 0) {
		printf("Error on call to getcontext\n");
		abort();
	}
}

context::context(void (*func_ptr)(void*), void* arg_ptr) :
		active(true), stack_del(true), continuations_closed(false), yield_bit(nullptr) {
	getcontext(this);
	uc_link = nullptr;
	uc_stack.ss_size = stack_size;
	uc_stack.ss_sp = stacks_alloc();
	++(nstacks());
	makecontext(this, reinterpret_cast<void (*)()>(&wrapper), 3, func_ptr, arg_ptr, this);
}

context::~context() {
	if (stack_del) {
		stacks_free(uc_stack.ss_sp);
		--(nstacks());
	}
}

void worker::main_context(std::reference_wrapper<worker>&& this_worker) {
	{
		std::lock_guard < std::mutex > lock(worker_map_mutex());
		worker_map().emplace(std::make_pair(pthread_self(), this_worker));
	}
	++(worker_count());
	while (!ready()) {
		std::this_thread::yield();
	}

	if (hwloc_set_cpubind(topology(), this_worker.get().cpuset, HWLOC_CPUBIND_THREAD)) {
		char *str;
		int error = errno;
		hwloc_bitmap_asprintf(&str, this_worker.get().cpuset);
		printf("Couldn't bind to cpuset %s: %s\n", str, strerror(error));
		free(str);
		abort();
	}

	this_worker.get().main_ctx = std::make_shared<context>();
	while (!finalize()) {
		this_worker.get().current_ctx = nullptr;
		hpx::this_thread::yield();
	}
	--(worker_count());
}

worker::worker(const std::size_t _proc_num, const hwloc_obj_t& obj) :
		thread(main_context, std::ref < worker > (*this)), proc_num(_proc_num) {
	cpuset = hwloc_bitmap_dup(obj->cpuset);
}

void worker::join() {
	thread.join();
}

//#define STACK_ALLOCATOR
static void* stacks_alloc() {
	void* ptr;
#ifdef STACK_ALLOCATOR
	std::unique_lock < std::mutex > lock(stacks_mutex());
	if (!stacks_avail().empty()) {
		ptr = stacks_avail().top();
		stacks_avail().pop();
		lock.unlock();
	} else {
		lock.unlock();
#endif
		ptr = new std::uint8_t[stack_size];
#ifdef STACK_ALLOCATOR
	}
#endif
	return ptr;
}

static void stacks_free(void* ptr) {
#ifndef STACK_ALLOCATOR
	delete[] reinterpret_cast<std::uint8_t*>(ptr);
#else
	std::lock_guard < std::mutex > lock(stacks_mutex());
	stacks_avail().push(ptr);
#endif
}

static void delayed_push(int i) {
	auto worker_it = worker_map().find(pthread_self());
	assert(worker_it != worker_map().end());
	worker& worker = worker_it->second.get();
	const auto ptr = worker.to_q_ctx;
	worker.to_q_ctx = nullptr;
	if (ptr != nullptr) {
		if (ptr->is_active()) {
			queue_push(i, ptr);
		}
	}

}

static bool pseudo_rand(float cutoff) {
	static std::atomic<std::size_t> chance_cnt(0);
	constexpr std::size_t chance_max = 100;
	constexpr std::size_t chance_inc = 47;
	chance_cnt += chance_inc;
	const auto r = float(chance_cnt % chance_max) / float(chance_max);
	return (r < cutoff);
}

static std::shared_ptr<context> queue_pop(int i) {
	int other_q;
	if (num_queues() > 1) {
		bool search_right = true;
		if (num_queues() > 2) {
			search_right = pseudo_rand(0.5);
		}
		if (search_right) {
			other_q = i + 1;
			if (other_q == int(num_queues())) {
				other_q = 0;
			}
		} else {
			other_q = i - 1;
			if (other_q == -1) {
				other_q = num_queues() - 1;
			}
		}
		const float chance = std::sqrt(std::max(0.0, 1.0 - q_thread_cnt(i) / std::max(1, int(q_thread_cnt(other_q)))));
		if (pseudo_rand(chance)) {
			i = other_q;
		}
	}
	std::shared_ptr<context> ptr;
	std::unique_lock < std::mutex > lock(queue_mutex(i));
	if (!queue(i).empty()) {
		ptr = queue(i).front();
		queue(i).pop_front();
		--q_thread_cnt(i);
	} else {
		ptr = nullptr;
	}
	return ptr;
}

static void queue_push(int i, const std::shared_ptr<context>& ptr) {
//	if (rand() % 10000 == 0) {
//		print_queues();
//	}
	++q_thread_cnt(i);
	std::lock_guard < std::mutex > lock(queue_mutex(i));
	queue(i).push_back(ptr);
}

void thread_finalize() {
	finalize() = true;
	for (auto& ptr : worker_ptr()) {
		ptr->join();
	}
}

#ifdef _GLIBCXX_DEBUG
#define SERIAL
#endif

void thread_initialize() {

	hwloc_topology_init(&(topology()));
	hwloc_topology_load(topology());

#ifndef SERIAL
	num_queues() = hwloc_get_nbobjs_by_type(topology(), HWLOC_OBJ_QUEUE);
#else
	num_queues() = 1;
#endif

#ifndef SERIAL
	std::size_t ncore = hwloc_get_nbobjs_by_type(topology(), HWLOC_OBJ_CORE);
	std::size_t nproc = hwloc_get_nbobjs_by_type(topology(), HWLOC_OBJ_PU);
//	std::size_t nproc = ncore;
	std::size_t pu_per_core = nproc / ncore;
#else
	std::size_t nproc = 1;
	std::size_t ncore = 1;
	std::size_t pu_per_core = 1;
#endif
	worker_ptr().resize(nproc);
	for (std::size_t ci = 0; ci != ncore; ++ci) {
#ifdef SERIAL
		hwloc_obj_t core = hwloc_get_obj_by_type(topology(), HWLOC_OBJ_CORE, ci);
#else
		hwloc_obj_t core = hwloc_get_obj_by_type(topology(), HWLOC_OBJ_MACHINE, 0);
#endif
		for (std::size_t pi = 0; pi != pu_per_core; ++pi) {
			const auto i = pu_per_core * ci + pi;
			worker_ptr()[i] = std::make_shared < worker > (i, core);
		}
	}
	while (worker_count() != nproc) {
		std::this_thread::yield();
	}
	ready() = true;
}

std::shared_ptr<context> thread_fork(void (*func_ptr)(void*), void* arg_ptr) {
	static std::atomic<std::size_t> next_q(0);
	const int q = next_q++ % num_queues();
	auto ctx_ptr = std::make_shared < context > (func_ptr, arg_ptr);
	queue_push(q, ctx_ptr);
	return ctx_ptr;
}

void check_mpi_return_code(int code, const char* file, int line) {
	if (code != MPI_SUCCESS) {
		printf("Internal MPI error - ");
		switch (code) {
		case MPI_ERR_COMM:
			printf("MPI_ERR_COMM");
			break;
		case MPI_ERR_RANK:
			printf("MPI_ERR_RANK");
			break;
		case MPI_ERR_TAG:
			printf("MPI_ERR_TAG");
			break;
		case MPI_ERR_PENDING:
			printf("MPI_ERR_PENDING");
			break;
		case MPI_ERR_IN_STATUS:
			printf("MPI_ERR_IN_STATUS");
			break;
		default:
			printf("unknown code %i ", code);
		}
		printf(" - from call in %s on line %i\n", file, line);
		exit (EXIT_FAILURE);
	}
}

}

thread::thread() {
}

bool thread::attempt_continuation(std::function<void()>& func) {
	return handle->add_continuation(func);
}

bool thread::joinable() const {
	return handle != nullptr;
}

thread::id thread::get_id() const {
	thread::id my_id;
	HPX_MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &(my_id.global)));
	my_id.local = reinterpret_cast<std::uintptr_t>(handle.get());
	return my_id;
}

thread::native_handle_type thread::native_handle() {
}

void thread::detach() {
	handle = nullptr;
}

void thread::join() {
	while (handle->is_active()) {
		this_thread::yield();
	}
	handle = nullptr;
}

void thread::swap(thread& other) {
	std::swap(handle, other.handle);
}

thread::id::id() :
		local(0), global(0) {
}

bool thread::id::operator==(const id& other) const {
	return (global == other.global) && (local == other.local);
}

bool thread::id::operator!=(const id& other) const {
	return !(operator==(other));
}

bool thread::id::operator<(const id& other) const {
	return (global < other.global) || ((global == other.global) && (local < other.local));
}

bool thread::id::operator>(const id& other) const {
	return !(operator<=(other));
}

bool thread::id::operator<=(const id& other) const {
	return !(operator>(other));
}

bool thread::id::operator>=(const id& other) const {
	return operator==(other) || operator>(other);
}

unsigned thread::hardware_concurrency() {
	return std::thread::hardware_concurrency();
}

namespace this_thread {

void yield(const std::shared_ptr<bool> yield_bit) {
	using namespace hpx::detail;
	auto worker_it = worker_map().find(pthread_self());
	if (worker_it != worker_map().end()) {
		ucontext_t *oucp, *ucp;
		bool swap_flag;
		std::size_t queue_num;
		{
			worker& w = worker_it->second.get();
			auto current_ptr = w.current_ctx;
			if (current_ptr != nullptr) {
				current_ptr->set_yield_bit(yield_bit);
			}
			queue_num = w.queue_num();
			std::shared_ptr<context> next_ptr;
			bool found_thread = false;
			do {
				next_ptr = queue_pop(queue_num);
				if (next_ptr == nullptr) {
					next_ptr = w.main_ctx;
				}
				if (next_ptr->is_active()) {
//					if (next_ptr->get_yield_bit()) {
						found_thread = true;
//					} else {
//						queue_push(queue_num, next_ptr);
//						found_thread = false;
//					}
				} else {
					found_thread = false;
				}
			} while (!found_thread);
			if (current_ptr != nullptr) {
				assert(w.to_q_ctx == nullptr);
				w.to_q_ctx = current_ptr;
			} else {
				current_ptr = w.main_ctx;
				w.current_ctx = nullptr;
				w.to_q_ctx = nullptr;
			}
			if (next_ptr.get() != w.main_ctx.get()) {
				w.current_ctx = next_ptr;
			} else {
				w.current_ctx = nullptr;
			}
			oucp = current_ptr.get();
			ucp = next_ptr.get();
			swap_flag = (next_ptr.get() != current_ptr.get());
		}

		if (swap_flag) {
			swapcontext(oucp, ucp);
		}

		delayed_push(queue_num);
	} else {
		std::this_thread::yield();
	}
}

thread::id get_id() {
	using namespace hpx::detail;
	thread::id my_id;
	auto worker_it = worker_map().find(pthread_self());
	worker& w = worker_it->second.get();
	HPX_MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &(my_id.global)));
	my_id.local = reinterpret_cast<std::uintptr_t>(w.current_ctx.get());
	return my_id;
}

namespace detail {

void set_exit_function(std::function<void()>&& func) {
	using namespace hpx::detail;
	thread::id my_id;
	auto worker_it = worker_map().find(pthread_self());
	worker& w = worker_it->second.get();
	w.current_ctx->add_exit_function(std::move(func));
}

}

}
}

namespace std {
size_t hash<hpx::thread::id>::operator()(const hpx::thread::id& id) const {
	hash<decltype(id.local)> local_hash;
	hash<decltype(id.global)> global_hash;
	return local_hash(id.local) ^ global_hash(id.global);
}

void swap(hpx::thread& t1, hpx::thread& t2) {
	t1.swap(t2);
}
}

