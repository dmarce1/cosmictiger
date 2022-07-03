#include <cosmictiger/fourier.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/vector.hpp>

static std::vector<vector<cmplx>> Y;
static vector<std::shared_ptr<spinlock_type>> mutexes;
static int N;
static int begin;
static std::vector<hpx::id_type> localities;
static int end;
static int span;
static int rank;
static int nranks;

vector<cmplx> fourier3d_transpose(int xbegin, int xend, int zi, vector<cmplx>);
void fourier3d_transpose_xz();
void fft1d(cmplx* Y, int N);
std::pair<std::vector<float>, std::vector<size_t>> fourier3d_get_power_spectrum();

HPX_PLAIN_ACTION(fourier3d_get_power_spectrum);
HPX_PLAIN_ACTION(fourier3d_initialize);
HPX_PLAIN_ACTION(fourier3d_destroy);
HPX_PLAIN_ACTION(fourier3d_transpose);
HPX_PLAIN_ACTION(fourier3d_accumulate);
HPX_PLAIN_ACTION(fourier3d_accumulate_real);
HPX_PLAIN_ACTION(fourier3d_transpose_xz);
HPX_PLAIN_ACTION(fourier3d_read);
HPX_PLAIN_ACTION(fourier3d_read_real);
HPX_PLAIN_ACTION(fourier3d_mirror);
HPX_PLAIN_ACTION(fourier3d_inv_execute);

int slab_to_rank(int xi) {
	return std::min((int) ((xi + int(N % nranks != 0)) * nranks / N), nranks - 1);
}

void fourier3d_mirror() {
	PRINT( "Mirroring\n");
	std::vector<hpx::future<void>> futs;
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_mirror_action>(localities[i]));
		}
	}
	for (int xi = begin; xi < end; xi++) {
		if (xi <= N / 2) {
			int xi0 = (N - xi) % N;
			auto future = hpx::async<fourier3d_read_action>(localities[slab_to_rank(xi0)], xi0, xi0 + 1, 0, N, 0, N);
			futs.push_back(future.then([xi](hpx::future<vector<cmplx>> fut) {
				auto data = fut.get();
				for( int yi = 0; yi < N; yi++) {
					for( int zi = 0; zi < N; zi++) {
						Y[xi-begin][yi*N+zi] = data[((N-yi)%N)*N+((N-zi)%N)].conj();
					}
				}
			}));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT( "Done Mirroring\n");
}

vector<cmplx> fourier3d_read(int xb, int xe, int yb, int ye, int zb, int ze) {
	vector<cmplx> data;
	int yspan = ye - yb;
	int zspan = ze - zb;
	data.resize(zspan * yspan * (xe - xb));
	std::vector<hpx::future<void>> futs;
	if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			auto fut = hpx::async<fourier3d_read_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze);
			futs.push_back(
					fut.then(
							[&data, this_xb, this_xe, yspan, zspan, xb, yb, ye, zb, ze](hpx::future<vector<cmplx>> fut) {
								auto rank_data = fut.get();
								std::vector<hpx::future<void>> these_futs;
								int nthreads = hardware_concurrency();
								for( int proc = 0; proc < nthreads; proc++) {
									const auto func = [xb,this_xb,proc,this_xe,nthreads,yb,ye,zb,ze,&data,&rank_data,zspan, yspan]() {
										for( int xi = this_xb + proc; xi < this_xe; xi+=nthreads) {
											for( int yi = yb; yi < ye; yi++) {
												for( int zi = zb; zi < ze; zi++) {
													data[zspan*yspan*(xi-xb)+zspan*(yi-yb)+(zi-zb)] = rank_data[zspan*yspan*(xi-this_xb)+zspan*(yi-yb)+(zi-zb)];
												}
											}
										}
									};
									these_futs.push_back(hpx::async(func));
								}
								hpx::wait_all(these_futs.begin(),these_futs.end());
							}));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [xb,proc,xe,nthreads,yb,ye,zb,ze,yspan,zspan,&data]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = Y[xi - begin][zspan * (yi - yb)
							+ (zi - zb)];
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

std::vector<float> fourier3d_read_real(int xb, int xe, int yb, int ye, int zb, int ze) {
	std::vector<float> data;
	int yspan = ye - yb;
	int zspan = ze - zb;
	data.resize(zspan * yspan * (xe - xb));
	std::vector<hpx::future<void>> futs;
	if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			auto fut = hpx::async<fourier3d_read_real_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze);
			futs.push_back(
					fut.then(
							[&data, this_xb, this_xe, yspan, zspan, xb, yb, ye, zb, ze](hpx::future<std::vector<float>> fut) {
								auto rank_data = fut.get();
								std::vector<hpx::future<void>> these_futs;
								int nthreads = hardware_concurrency();
								for( int proc = 0; proc < nthreads; proc++) {
									const auto func = [xb,this_xb,proc,this_xe,nthreads,yb,ye,zb,ze,&data,&rank_data,zspan, yspan]() {
										for( int xi = this_xb + proc; xi < this_xe; xi+=nthreads) {
											for( int yi = yb; yi < ye; yi++) {
												for( int zi = zb; zi < ze; zi++) {
													data[zspan*yspan*(xi-xb)+zspan*(yi-yb)+(zi-zb)] = rank_data[zspan*yspan*(xi-this_xb)+zspan*(yi-yb)+(zi-zb)];
												}
											}
										}
									};
									these_futs.push_back(hpx::async(func));
								}
								hpx::wait_all(these_futs.begin(),these_futs.end());
							}));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [xb,proc,xe,nthreads,yb,ye,zb,ze,yspan,zspan,&data]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = Y[xi - begin][zspan * (yi - yb)
							+ (zi - zb)].real();
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

void fourier3d_accumulate(int xb, int xe, int yb, int ye, int zb, int ze, vector<cmplx> data) {
	const int xspan = xe - xb;
	const int yspan = ye - yb;
	const int zspan = ze - zb;
	std::vector<hpx::future<void>> futs;
	int max_nthreads = hardware_concurrency();
	int nthreads;
	if (xb < 0) {
		int xspan1 = xe - 0;
		int xspan2 = 0 - xb;
		vector<cmplx> data1(xspan1 * yspan * zspan);
		vector<cmplx> data2(xspan2 * yspan * zspan);
		nthreads = std::min(max_nthreads, yspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([xb,xe,yb,ye,zb,ze,xspan1,xspan2,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
				for (int i = 0; i < xe; i++) {
					for (int j = yb + proc; j < ye; j+= nthreads) {
						for (int k = zb; k < ze; k++) {
							data1[i * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}
				}
				for (int i = xb; i < 0; i++) {
					for (int j = yb + proc; j < ye; j+= nthreads) {
						for (int k = zb; k < ze; k++) {
							data2[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}}
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate(0, xe, yb, ye, zb, ze, std::move(data1));
		fourier3d_accumulate(xb + N, N, yb, ye, zb, ze, std::move(data2));
	} else if (xe > N) {
		int xspan1 = N - xb;
		int xspan2 = xe - N;
		vector<cmplx> data1(xspan1 * yspan * zspan);
		vector<cmplx> data2(xspan2 * yspan * zspan);
		nthreads = std::min(max_nthreads, yspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([xb,xe,yb,ye,zb,ze,xspan1,xspan2,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
				for (int i = xb; i < N; i++) {
					for (int j = yb + proc; j < ye; j+=nthreads) {
						for (int k = zb; k < ze; k++) {
							data1[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}
				}
				for (int i = N; i < xe; i++) {
					for (int j = yb + proc; j < ye; j+=nthreads) {
						for (int k = zb; k < ze; k++) {
							data2[(i - N) * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}
				}
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate(xb, N, yb, ye, zb, ze, std::move(data1));
		fourier3d_accumulate(0, xe - N, yb, ye, zb, ze, std::move(data2));
	} else if (yb < 0) {
		int yspan1 = ye - 0;
		int yspan2 = 0 - yb;
		vector<cmplx> data1(xspan * yspan1 * zspan);
		vector<cmplx> data2(xspan * yspan2 * zspan);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,yspan1,yspan2,xspan,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb+proc; i < xe; i+=nthreads) {
									for (int j = 0; j < ye; j++) {
										for (int k = zb; k < ze; k++) {
											data1[(i - xb) * yspan1 * zspan + j * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
									for (int j = yb; j < 0; j++) {
										for (int k = zb; k < ze; k++) {
											data2[(i - xb) * yspan2 * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate(xb, xe, 0, ye, zb, ze, std::move(data1));
		fourier3d_accumulate(xb, xe, yb + N, N, zb, ze, std::move(data2));
	} else if (ye > N) {
		int yspan1 = N - yb;
		int yspan2 = ye - N;
		vector<cmplx> data1(xspan * yspan1 * zspan);
		vector<cmplx> data2(xspan * yspan2 * zspan);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,yspan1,yspan2,xspan,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb + proc; i < xe; i+=nthreads) {
									for (int j = yb; j < N; j++) {
										for (int k = zb; k < ze; k++) {
											data1[(i - xb) * yspan1 * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
									for (int j = N; j < ye; j++) {
										for (int k = zb; k < ze; k++) {
											data2[(i - xb) * yspan2 * zspan + (j - N) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate(xb, xe, yb, N, zb, ze, std::move(data1));
		fourier3d_accumulate(xb, xe, 0, ye - N, zb, ze, std::move(data2));
	} else if (zb < 0) {
		int zspan1 = ze - 0;
		int zspan2 = 0 - zb;
		vector<cmplx> data1(xspan * yspan * zspan1);
		vector<cmplx> data2(xspan * yspan * zspan2);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,xspan,zspan1,zspan2,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb+proc; i < xe; i+=nthreads) {
									for (int j = yb; j < ye; j++) {
										for (int k = 0; k < ze; k++) {
											data1[(i - xb) * yspan * zspan1 + j * zspan1 + k] = data[(i - xb) * yspan * zspan + (j - yb) * zspan
											+ (k - zb)];
										}
										for (int k = 0; k < ze; k++) {
											data2[(i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate(xb, xe, yb, ye, 0, ze, std::move(data1));
		fourier3d_accumulate(xb, xe, yb, ye, zb + N, N, std::move(data2));
	} else if (ze > N) {
		int zspan1 = N - zb;
		int zspan2 = ze - N;
		vector<cmplx> data1(xspan * yspan * zspan1);
		vector<cmplx> data2(xspan * yspan * zspan2);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,xspan,yspan,zspan1,zspan2,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb+proc; i < xe; i+=nthreads) {
									for (int j = yb; j < ye; j++) {
										for (int k = zb; k < N; k++) {
											data1[(i - xb) * yspan * zspan1 + j * zspan1 + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
										for (int k = N; k < ze; k++) {
											data2[(i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - N)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate(xb, xe, yb, ye, zb, N, std::move(data1));
		fourier3d_accumulate(xb, xe, yb, ye, 0, ze - N, std::move(data2));
	} else if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			vector<cmplx> rank_data(this_xspan * yspan * zspan);
			int nthreads = hardware_concurrency();
			std::vector<hpx::future<void>> these_futs;
			for (int proc = 0; proc < nthreads; proc++) {
				const auto func = [this_xb,this_xe,proc,nthreads,xb,yb,ye,zb,ze,zspan,yspan,&data,&rank_data]() {
					for (int xi = this_xb + proc; xi < this_xe; xi+= nthreads) {
						for (int yi = yb; yi < ye; yi++) {
							for (int zi = zb; zi < ze; zi++) {
								rank_data[zspan * yspan * (xi - this_xb) + zspan * (yi - yb) + (zi - zb)] = data[zspan * yspan
								* (xi - xb) + yspan * (yi - yb) + (zi - zb)];
							}
						}
					}
				};
				these_futs.push_back(hpx::async(func));
			}
			hpx::wait_all(these_futs.begin(), these_futs.end());
			futs.push_back(
					hpx::async<fourier3d_accumulate_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze,
							std::move(rank_data)));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [proc,nthreads,xb,xe,yb,ye,zb,ze,&data,zspan,yspan]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					std::lock_guard<spinlock_type> lock(*mutexes[xi - xb]);
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							Y[xi - xb][N * (yi - yb) + (zi - zb)] += data[zspan * yspan * (xi - xb) + zspan * (yi - yb)
							+ (zi - zb)];
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_accumulate_real(int xb, int xe, int yb, int ye, int zb, int ze, vector<float> data) {
	const int xspan = xe - xb;
	const int yspan = ye - yb;
	const int zspan = ze - zb;
	std::vector<hpx::future<void>> futs;
	int max_nthreads = hardware_concurrency();
	int nthreads;
	if (xb < 0) {
		int xspan1 = xe - 0;
		int xspan2 = 0 - xb;
		vector<float> data1(xspan1 * yspan * zspan);
		vector<float> data2(xspan2 * yspan * zspan);
		nthreads = std::min(max_nthreads, yspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([xb,xe,yb,ye,zb,ze,xspan1,xspan2,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
				for (int i = 0; i < xe; i++) {
					for (int j = yb + proc; j < ye; j+= nthreads) {
						for (int k = zb; k < ze; k++) {
							data1[i * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}
				}
				for (int i = xb; i < 0; i++) {
					for (int j = yb + proc; j < ye; j+= nthreads) {
						for (int k = zb; k < ze; k++) {
							data2[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}}
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate_real(0, xe, yb, ye, zb, ze, std::move(data1));
		fourier3d_accumulate_real(xb + N, N, yb, ye, zb, ze, std::move(data2));
	} else if (xe > N) {
		int xspan1 = N - xb;
		int xspan2 = xe - N;
		vector<float> data1(xspan1 * yspan * zspan);
		vector<float> data2(xspan2 * yspan * zspan);
		nthreads = std::min(max_nthreads, yspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([xb,xe,yb,ye,zb,ze,xspan1,xspan2,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
				for (int i = xb; i < N; i++) {
					for (int j = yb + proc; j < ye; j+=nthreads) {
						for (int k = zb; k < ze; k++) {
							data1[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}
				}
				for (int i = N; i < xe; i++) {
					for (int j = yb + proc; j < ye; j+=nthreads) {
						for (int k = zb; k < ze; k++) {
							data2[(i - N) * yspan * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
							+ (j - yb) * zspan + (k - zb)];
						}
					}
				}
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate_real(xb, N, yb, ye, zb, ze, std::move(data1));
		fourier3d_accumulate_real(0, xe - N, yb, ye, zb, ze, std::move(data2));
	} else if (yb < 0) {
		int yspan1 = ye - 0;
		int yspan2 = 0 - yb;
		vector<float> data1(xspan * yspan1 * zspan);
		vector<float> data2(xspan * yspan2 * zspan);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,yspan1,yspan2,xspan,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb+proc; i < xe; i+=nthreads) {
									for (int j = 0; j < ye; j++) {
										for (int k = zb; k < ze; k++) {
											data1[(i - xb) * yspan1 * zspan + j * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
									for (int j = yb; j < 0; j++) {
										for (int k = zb; k < ze; k++) {
											data2[(i - xb) * yspan2 * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate_real(xb, xe, 0, ye, zb, ze, std::move(data1));
		fourier3d_accumulate_real(xb, xe, yb + N, N, zb, ze, std::move(data2));
	} else if (ye > N) {
		int yspan1 = N - yb;
		int yspan2 = ye - N;
		vector<float> data1(xspan * yspan1 * zspan);
		vector<float> data2(xspan * yspan2 * zspan);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,yspan1,yspan2,xspan,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb + proc; i < xe; i+=nthreads) {
									for (int j = yb; j < N; j++) {
										for (int k = zb; k < ze; k++) {
											data1[(i - xb) * yspan1 * zspan + (j - yb) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
									for (int j = N; j < ye; j++) {
										for (int k = zb; k < ze; k++) {
											data2[(i - xb) * yspan2 * zspan + (j - N) * zspan + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate_real(xb, xe, yb, N, zb, ze, std::move(data1));
		fourier3d_accumulate_real(xb, xe, 0, ye - N, zb, ze, std::move(data2));
	} else if (zb < 0) {
		int zspan1 = ze - 0;
		int zspan2 = 0 - zb;
		vector<float> data1(xspan * yspan * zspan1);
		vector<float> data2(xspan * yspan * zspan2);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async(
							[xb,xe,yb,ye,zb,ze,xspan,zspan1,zspan2,yspan,zspan,&data,&data1,&data2,proc,nthreads]() {
								for (int i = xb+proc; i < xe; i+=nthreads) {
									for (int j = yb; j < ye; j++) {
										for (int k = 0; k < ze; k++) {
											data1[(i - xb) * yspan * zspan1 + j * zspan1 + k] = data[(i - xb) * yspan * zspan + (j - yb) * zspan
											+ (k - zb)];
										}
										for (int k = 0; k < ze; k++) {
											data2[(i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - zb)] = data[(i - xb) * yspan * zspan
											+ (j - yb) * zspan + (k - zb)];
										}
									}
								}
							}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate_real(xb, xe, yb, ye, 0, ze, std::move(data1));
		fourier3d_accumulate_real(xb, xe, yb, ye, zb + N, N, std::move(data2));
	} else if (ze > N) {
		int zspan1 = N - zb;
		int zspan2 = ze - N;
		vector<float> data1(xspan * yspan * zspan1);
		vector<float> data2(xspan * yspan * zspan2);
		nthreads = std::min(max_nthreads, xspan);
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(
					hpx::async([xb,xe,yb,ye,zb,ze,xspan,yspan,zspan1,zspan2,zspan,&data,&data1,&data2,proc,nthreads]() {
						for (int i = xb+proc; i < xe; i+=nthreads) {
							for (int j = yb; j < ye; j++) {
								for (int k = zb; k < N; k++) {
									int i0 = (i - xb) * yspan * zspan1 + j * zspan1 + (k - zb);
									int i1 = (i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb);
									data1[i0] = data[i1];
								}
								for (int k = N; k < ze; k++) {
									int i0 = (i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - N);
									int i1 = (i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb);
									data2[i0] = data[i1];
								}
							}
						}
					}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		data = decltype(data)();
		fourier3d_accumulate_real(xb, xe, yb, ye, zb, N, std::move(data1));
		fourier3d_accumulate_real(xb, xe, yb, ye, 0, ze - N, std::move(data2));
	} else if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			vector<float> rank_data(this_xspan * yspan * zspan);
			int nthreads = hardware_concurrency();
			std::vector<hpx::future<void>> these_futs;
			for (int proc = 0; proc < nthreads; proc++) {
				const auto func = [this_xb,this_xe,proc,nthreads,xb,yb,ye,zb,ze,zspan,yspan,&data,&rank_data]() {
					for (int xi = this_xb + proc; xi < this_xe; xi+= nthreads) {
						for (int yi = yb; yi < ye; yi++) {
							for (int zi = zb; zi < ze; zi++) {
								int i0 = zspan * yspan * (xi - this_xb) + zspan * (yi - yb) + (zi - zb);
								int i1 = zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb);
								assert(i0>=0);
								assert(i0<rank_data.size());
								assert(i1>=0);
								assert(i1<data.size());
								rank_data[i0] = data[i1];
							}
						}
					}
				};
				these_futs.push_back(hpx::async(func));
			}
			hpx::wait_all(these_futs.begin(), these_futs.end());
			futs.push_back(
					hpx::async<fourier3d_accumulate_real_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze,
							std::move(rank_data)));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [proc,nthreads,xb,xe,yb,ye,zb,ze,&data,zspan,yspan]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					std::lock_guard<spinlock_type> lock(*mutexes[xi - xb]);
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							Y[xi - xb][N * (yi - yb) + (zi - zb)].real() += data[zspan * yspan * (xi - xb) + zspan * (yi - yb)
							+ (zi - zb)];
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_do2dpart() {
	for (int i = 0; i < span; i++) {
		fft2d(Y[i].data(), N);
	}
}

void fourier3d_do1dpart() {
	for (int i = 0; i < span; i++) {
		fft1d(Y[i].data(), N);
	}
}

HPX_PLAIN_ACTION(fourier3d_do1dpart);
HPX_PLAIN_ACTION(fourier3d_do2dpart);

std::vector<float> fourier3d_power_spectrum() {
	std::vector<hpx::future<std::pair<std::vector<float>, std::vector<size_t>>> >futs;
	for( int i = 0; i < nranks; i++) {
		futs.push_back(hpx::async<fourier3d_get_power_spectrum_action>(localities[i]));
	}
	std::vector<float> power;
	std::vector<size_t> counts;
	int maxN = N / 2;
	power.resize(maxN, 0.0f);
	counts.resize(maxN, 0);
	for( auto& f : futs ) {
		auto data = f.get();
		auto this_power = std::move(data.first);
		auto this_counts = std::move(data.second);
		for( int i = 0; i < maxN; i++) {
			power[i] += this_power[i];
			counts[i] += this_counts[i];
		}
	}
	for( int i = 0; i < maxN; i++) {
		if( counts[i] ) {
			power[i] /= counts[i];
		}
	}
	return power;
}

std::pair<std::vector<float>, std::vector<size_t>> fourier3d_get_power_spectrum() {
	std::vector<float> power;
	std::vector<size_t> counts;
	int maxN = N / 2;
	power.resize(maxN, 0.0f);
	counts.resize(maxN, 0);
	const int nthreads = hardware_concurrency();
	static mutex_type mutex;
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([maxN,proc,nthreads,&counts,&power]() {
			std::vector<float> this_power;
			std::vector<size_t> this_counts;
			this_power.resize(maxN, 0.0f);
			this_counts.resize(maxN, 0);
			for (int yi = proc; yi < N; yi+=nthreads) {
				const int yi0 = yi < N / 2 ? yi : yi - N;
				for (int xi = begin; xi < end; xi++) {
					const int xi0 = xi < N / 2 ? xi : xi - N;
					for (int zi = 0; zi < N; zi++) {
						const int zi0 = zi < N / 2 ? zi : zi - N;
						const int i2 = xi0 * xi0 + yi0 * yi0 + zi0 * zi0;
						const int i = int(std::sqrt(i2));
						if( i < maxN ) {
							assert(Y.size() > xi - begin);
							const float mag = Y[xi-begin][N*yi+zi].norm();
							this_power[i] += mag;
							this_counts[i]++;
						}
					}
				}
			}
			std::lock_guard<mutex_type> lock(mutex);
			for( int i = 0; i < maxN; i++) {
				power[i] += this_power[i];
				counts[i] += this_counts[i];
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::make_pair(std::move(power), std::move(counts));
}

void fourier3d_execute() {
	PRINT("Executing fourier\n");
	std::vector<hpx::future<void>> futs;
	PRINT("doing 2d\n");
	for (int i = 0; i < nranks; i++) {
		futs.push_back(hpx::async<fourier3d_do2dpart_action>(localities[i]));
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("transposing\n");
	fourier3d_transpose_xz();
	futs.resize(0);
	for (int i = 0; i < nranks; i++) {
		futs.push_back(hpx::async<fourier3d_do1dpart_action>(localities[i]));
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("transposing\n");
	fourier3d_transpose_xz();
	PRINT("Done executing fourier\n");
}

void fourier3d_inv_execute() {
	std::vector<hpx::future<void>> futs;
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_inv_execute_action>(localities[i]));
		}
	}
	const float factor = 1.0f / (N * N * N);
	for (int i = 0; i < span; i++) {
		for (int j = 0; j < N * N; j++) {
			Y[i][j].real() *= factor;
			Y[i][j].imag() *= factor;
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	if (rank == 0) {
		fourier3d_execute();
	}
}

void fourier3d_initialize(int N_) {
	std::vector<hpx::future<void>> futs;
	rank = hpx_rank();
	nranks = hpx_size();
	localities = hpx_localities();
	if (rank == 0) {
		fourier3d_destroy();
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_initialize_action>(localities[i], N_));
		}
	}
	N = N_;
	begin = rank * N / nranks;
	end = (rank + 1) * N / nranks;
	span = end - begin;
	Y.resize(span);
	mutexes.resize(span);
	for (int i = 0; i < span; i++) {
		mutexes[i] = std::make_shared<spinlock_type>();
		Y[i].resize(N * N);
		for (int j = 0; j < N * N; j++) {
			Y[i][j].real() = Y[i][j].imag() = 0.0;
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_destroy() {
	std::vector<hpx::future<void>> futs;
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_destroy_action>(localities[i]));
		}
	}
	Y = decltype(Y)();
	mutexes = decltype(mutexes)();
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_transpose_xz() {
	std::vector<hpx::future<void>> futs;
	rank = hpx_rank();
	nranks = hpx_size();
	localities = hpx_localities();
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_transpose_xz_action>(localities[i]));
		}
	}
	for (int other = rank; other < nranks; other++) {
		for (int xi = begin; xi < end; xi++) {
			vector<cmplx> data;
			const int zend = (other + 1) * N / nranks;
			const int zbegin = other * N / nranks + ((other == rank) ? (xi - begin) : 0);
			const int zspan = zend - zbegin;
			data.resize(N * zspan);
			for (int j = 0; j < N; j++) {
				for (int k = zbegin; k < zend; k++) {
					data[j * zspan + k - zbegin] = Y[xi - begin][j * N + k];
				}
			}
			auto future = hpx::async<fourier3d_transpose_action>(localities[other], zbegin, zend, xi, std::move(data));
			futs.push_back(future.then([zspan,zbegin,zend,xi](hpx::future<vector<cmplx>> fut) {
				auto data = fut.get();
				for (int j = 0; j < N; j++) {
					for (int k = zbegin; k < zend; k++) {
						Y[xi-begin][j * N + k] = data[j * zspan + k - zbegin];
					}
				}
			}));
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

vector<cmplx> fourier3d_transpose(int xbegin, int xend, int zi, vector<cmplx> data) {
	const int xspan = xend - xbegin;
	for (int xi = 0; xi < xspan; xi++) {
		for (int yi = 0; yi < N; yi++) {
			auto& a = data[yi * xspan + xi];
			auto& b = Y[xi + xbegin - begin][yi * N + zi];
			swap(a, b);
		}
	}
	return std::move(data);
}
