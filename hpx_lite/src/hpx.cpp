/*
 * hpx.cpp
 *
 *  Created on: Sep 22, 2015
 *      Author: dmarce1
 */

#include "hpx/hpx_lite.hpp"
#include <hwloc.h>
#include <list>
#include <unistd.h>
#include <unordered_map>

static bool exit_signal = false;

namespace hpx {

namespace detail {

typedef std::pair<std::string, naked_action_type> action_dir_entry_type;
typedef std::unordered_map<std::string, naked_action_type> action_dir_type;

static action_dir_type& action_dir() {
	static action_dir_type a;
	return a;
}

static std::mutex& action_dir_mutex() {
	static std::mutex a;
	return a;
}

void register_action_name(const std::string& strname, naked_action_type fptr) {
	std::lock_guard < std::mutex > lock(action_dir_mutex());
	action_dir().insert(std::make_pair(strname, fptr));
}

naked_action_type lookup_action(const std::string& strname) {
	return action_dir()[strname];
}

int& mpi_comm_rank() {
	static int r;
	return r;
}
int& mpi_comm_size() {
	static int r;
	return r;
}
}
}

extern int hpx_main(int argc, char* argv[]);

namespace hpx {
namespace detail {
struct message_type {
	int target_rank;
	obuffer_type buffer;
	MPI_Request mpi_request;
	int message_size;
	bool sent;
};

typedef std::list<std::shared_ptr<message_type>> message_queue_type;

std::uintptr_t promise_table_generate_future(hpx::future<void>& future) {
	auto pointer = new hpx::promise<void>();
	future = pointer->get_future();
	return reinterpret_cast<std::uintptr_t>(pointer);
}

void promise_table_satisfy_void_promise(obuffer_type data) {
	std::uintptr_t index;
	data >> index;
	auto pointer = reinterpret_cast<hpx::promise<void>*>(index);
	pointer->set_value();
	delete pointer;
}

static void handle_incoming_message(int, obuffer_type);
static int handle_outgoing_message(std::shared_ptr<message_type>);

static message_queue_type& message_queue() {
	static message_queue_type a;
	return a;
}

static hpx::lcos::local::mutex& message_queue_mutex() {
	static hpx::lcos::local::mutex a;
	return a;
}

static std::atomic<bool>& main_exit_signal() {
	static std::atomic<bool> a(false);
	return a;
}

}
}

static void local_finalize();

HPX_PLAIN_ACTION(local_finalize, finalize_action);


int main(int argc, char* argv[]) {
	hwloc_topology_t topology;
	hwloc_topology_init(&(topology));
	hwloc_topology_load(topology);

	const auto obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_SOCKET, 0);

	if (hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD)) {
		char *str;
		int error = errno;
		hwloc_bitmap_asprintf(&str, obj->cpuset);
		printf("Couldn't bind to cpuset %s: %s\n", str, strerror(error));
		free(str);
		abort();
	}

	int rank, provided, rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided != MPI_THREAD_FUNNELED && provided != MPI_THREAD_MULTIPLE) {
		printf("MPI threading insufficient\n");
		rc = EXIT_FAILURE;
	} else {
		hpx::detail::thread_initialize();
		MPI_Comm_rank(MPI_COMM_WORLD, &hpx::detail::mpi_comm_rank());
		MPI_Comm_size(MPI_COMM_WORLD, &hpx::detail::mpi_comm_size());
		rank = hpx::detail::mpi_comm_rank();
		hpx::thread thd;
		if (rank == 0) {
			thd = hpx::thread([](int a, char* b[]) {
				hpx_main(a, b);
				hpx::detail::main_exit_signal() = true;
			}, argc, argv);
		}
		if (hpx::detail::mpi_comm_size() > 1) {
			if (rank == 0) {
				thd.detach();
			}
			hpx::detail::server(argc, argv);
		} else {
			thd.join();
		}
		rc = EXIT_SUCCESS;
	}
	MPI_Finalize();
	hpx::detail::thread_finalize();
	return rc;
}

static void local_finalize() {
	hpx::thread([]() {
		sleep(1);
		hpx::detail::set_exit_signal();
	}).detach();

}

namespace hpx {

int finalize() {
	hpx::thread([]() {
		while(!hpx::detail::main_exit_signal()) {
			hpx::this_thread::yield();
		}
		int size;
		size = hpx::detail::mpi_comm_size();
		for (int i = size - 1; i > 0; i--) {
			hpx::id_type id;
			id.set_rank(i);
			hpx::async<finalize_action>(id).get();
		}
		hpx::detail::set_exit_signal();
	}).detach();

	return EXIT_SUCCESS;
}

int get_locality_id() {
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	return rank;
}

std::vector<hpx::id_type> find_all_localities() {
	int size;
	size = hpx::detail::mpi_comm_size();
	std::vector<hpx::id_type> localities(size);
	for (int i = 0; i < size; ++i) {
		localities[i].set_rank(i);
	}
	return localities;
}

namespace detail {

void set_exit_signal() {
	exit_signal = true;
}

void remote_action(int target, obuffer_type&& buffer) {
	auto message = std::make_shared<message_type>();
	message->target_rank = target;
	message->sent = false;
	message->buffer = std::move(buffer);
	std::lock_guard<hpx::lcos::local::mutex> lock(message_queue_mutex());
	message_queue().push_back(message);
}

int server(int argc, char* argv[]) {
	obuffer_type buffer;
	bool found_stuff;
	do {
		found_stuff = false;
		int rc;
		MPI_Status stat;
		int flag;
		do {
			flag = 0;
			int count, src;
			rc = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &stat);
			HPX_MPI_CALL(rc);
			if (flag) {
				found_stuff = true;
				src = stat.MPI_SOURCE;
				count = stat.MPI_TAG;
				buffer = obuffer_type(count);
				rc = MPI_Recv(buffer.data(), count, MPI_BYTE, src, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				HPX_MPI_CALL(rc);
				auto buffer_ptr = std::make_shared < obuffer_type > (std::move(buffer));
				hpx::thread([=]() {
					handle_incoming_message(src, *buffer_ptr);
				}).detach();
			}
		} while (flag && !exit_signal);
		std::unique_lock<hpx::lcos::local::mutex> lock(message_queue_mutex());
		auto i = message_queue().begin();
		while (i != message_queue().end() && !exit_signal) {
			found_stuff = true;
			lock.unlock();
			int rc = handle_outgoing_message(*i);
			if (rc == +1) {
				lock.lock();
				i = message_queue().erase(i);
			} else {
				auto tmp = *i;
				++i;
				if (rc == -1) {
					found_stuff = true;
				}
				lock.lock();
			}
		}
		if (!found_stuff) {
//			hpx::this_thread::yield();
		}
	} while (found_stuff || !exit_signal);
	return EXIT_SUCCESS;
}

static void handle_incoming_message(int return_rank, obuffer_type message) {
	ibuffer_type buffer;
	invokation_type itype;
	message >> itype;
	if (itype != PROMISE) {
		naked_action_type func_ptr;
		if (itype == NAME) {
			std::string function_name;
			message >> function_name;
			func_ptr = lookup_action(function_name);
		} else if (itype == ADDRESS) {
			std::uintptr_t function_name;
			message >> function_name;
			func_ptr = reinterpret_cast<naked_action_type>(function_name);
		} else {
			assert(false);
			func_ptr = nullptr;
		}
		std::uintptr_t index;
		std::uintptr_t promise_function;
		message >> index;
		message >> promise_function;
		buffer << hpx::detail::invokation_type(PROMISE);
		buffer << promise_function;
		buffer << index;
		(*func_ptr)(buffer, std::move(message));
		hpx::detail::remote_action(return_rank, std::move(buffer));
	} else {
		std::uintptr_t ptrint;
		message >> ptrint;
		auto func_ptr = reinterpret_cast<promise_action_type>(ptrint);
		(*func_ptr)(std::move(message));
	}
}

void pack_args(ibuffer_type& buffer) {
}

static int handle_outgoing_message(std::shared_ptr<message_type> message) {
	int rc, mpi_rc;
	int flag;
	if (!message->sent) {
		int count = message->buffer.size();
		int dest = message->target_rank;
		void* buffer = message->buffer.data();
		MPI_Request* const request = &(message->mpi_request);
		mpi_rc = MPI_Isend(buffer, count, MPI_BYTE, dest, count, MPI_COMM_WORLD, request);
		HPX_MPI_CALL(mpi_rc);
		message->sent = true;
		rc = -1;
	} else {
		MPI_Request* request = &(message->mpi_request);
		mpi_rc = MPI_Test(request, &flag, MPI_STATUS_IGNORE);
		HPX_MPI_CALL(mpi_rc);
		if (!flag) {
			rc = 0;
		} else {
			rc = +1;
		}
	}
	return rc;
}

}

namespace naming {

int get_locality_id_from_gid(const gid_type& gid) {
	return gid.get_rank();
}

}

bool id_type::operator==(const id_type& other) const {
	return !(id_type::operator!=(other));
}

bool id_type::operator!=(const id_type& other) const {
	bool rc;
	if (rank != other.rank) {
		rc = true;
	} else if (address == nullptr) {
		rc = other.address != nullptr;
	} else if (other.address == nullptr) {
		rc = address != nullptr;
	} else if (*address != *(other.address)) {
		rc = true;
	} else {
		rc = false;
	}
	return rc;

}

id_type find_here() {
	id_type id;
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	id.set_rank(rank);
	return id;

}
}

HPX_REGISTER_PLAIN_ACTION(action_add_ref_count);

