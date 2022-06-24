template<unsigned SIZE>
class lockfree_queue {
	using itype = unsigned long long int;
	unsigned qin;
	unsigned qout;
	array<char*, SIZE> q;
#ifndef __CUDA_ARCH__
	static unsigned atomic_max(unsigned* address, unsigned val) {
		unsigned assumed;
		unsigned old = *address;
		do {
			assumed = old;
			old = __sync_val_compare_and_swap(address, assumed, std::max(val, assumed));
		} while (assumed != old);
		return old;
	}
#endif
public:
	lockfree_queue() {
		for (unsigned i = 0; i < SIZE; i++) {
			q[i] = nullptr;
		}
		qin = qout = 0;
	}
#ifdef __CUDA_ARCH__
	__device__
	inline bool push(char* ptr) {
		auto in = qin;
		const auto& out = qout;
		if (in - out >= SIZE) {
			return false;
		}
		while (atomicCAS((itype*) &q[in % SIZE], (itype) 0, (itype) ptr) != 0) {
			in++;
			if (in - out >= SIZE) {
				return false;
			}
		}
		in++;
		atomicMax(&qin, in);
		return true;
	}
	__device__
	inline char* pop() {
		const auto& in = qin;
		auto out = qout;
		if (out >= in) {
			return nullptr;
		}
		char* ptr;
		while ((ptr = (char*) atomicExch((itype*) &q[out % SIZE], (itype) 0)) == nullptr) {
			if (out >= in) {
				return nullptr;
			}
			out++;
		}
		out++;
		atomicMax(&qout, out);
		return ptr;

	}
#else
	inline bool push(char* ptr) {
		auto in = qin;
		const auto& out = qout;
		if (in - out >= SIZE) {
			return false;
		}
		while (__sync_val_compare_and_swap((itype*) &q[in % SIZE], (itype) 0, (itype) ptr) != 0) {
			in++;
			if (in - out >= SIZE) {
				return false;
			}
		}
		in++;
		atomic_max(&qin, in);
		return true;
	}
	inline char* pop() {
		const auto& in = qin;
		auto out = qout;
		if (out >= in) {
			return nullptr;
		}
		char* ptr = q[out % SIZE];
		while (__sync_val_compare_and_swap((itype*) &q[out % SIZE], (itype) ptr, (itype) 0) != (itype) ptr) {
			if (out >= in) {
				return nullptr;
			}
			out++;
			ptr = q[out % SIZE];
		}
		out++;
		atomic_max(&qout, out);
		return ptr;
	}
#endif
};

