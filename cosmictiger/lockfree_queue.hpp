
template<int SIZE>
class lockfree_queue {
	using itype = unsigned long long int;
	int qin;
	int qout;
	array<char*, SIZE> q;
public:
	lockfree_queue() {
		for (int i = 0; i < SIZE; i++) {
			q[i] = nullptr;
		}
		qin = qout = 0;
	}
#ifdef __CUDACC__
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
#endif
};

