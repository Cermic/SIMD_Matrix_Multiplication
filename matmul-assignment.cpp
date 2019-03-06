#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <immintrin.h>

// $CXX -03 -mavx matmul_assignment.cpp

#if (!defined(_MSC_VER))
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#define SZ (1 << 2) // (1 << 10) == 1024

struct mat {
  float *data;
  const size_t sz;
  bool operator==(const mat &rhs) const {
    return !std::memcmp(data,rhs.data,sizeof(sz*sz*sizeof(data[0])));
  }
};

struct matd {
	double *data;
	const size_t sz;
	bool operator==(const matd &rhs) const {
		return !std::memcmp(data, rhs.data, sizeof(sz*sz * sizeof(data[0])));
	}
};

//template<typename U>
//struct U {
//	U *data;
//	const size_t sz;
//	bool operator==(const U &rhs) const {
//		return !std::memcmp(data, rhs.data, sizeof(sz*sz * sizeof(data[0]))); // How do you setup this type to be based on a generic one that's passed in at compile time?
//	}
//};

void matmul(mat &mres, const mat &m1, const mat &m2)
{
  for (int i = 0; i < mres.sz; i++) {
    for (int j = 0; j < mres.sz; j++) {
      mres.data[i*mres.sz+j] = 0;
      for (int k = 0; k < mres.sz; k++) {
        mres.data[i*mres.sz+j] += m1.data[i*mres.sz+k] * m2.data[k*mres.sz+j];
      }
    }
  }
}

void SIMD_matmul (mat &mres, const mat &m1, const mat &m2)
{
	for (int i = 0; i < mres.sz; i++)
	{
		// Collect the rows of the matrix
		__m128 vx = _mm_broadcast_ss(&m1.data[i*mres.sz]);
		__m128 vy = _mm_broadcast_ss(&m1.data[i*mres.sz + 1]);
		__m128 vz = _mm_broadcast_ss(&m1.data[i*mres.sz + 2]);
		__m128 vw = _mm_broadcast_ss(&m1.data[i*mres.sz + 3]);

		// Perform the multiplication on the rows
		vx = _mm_mul_ps(vx, _mm_load_ps(&m2.data[0 * mres.sz]));
		vy = _mm_mul_ps(vy, _mm_load_ps(&m2.data[mres.sz]));
		vz = _mm_mul_ps(vz, _mm_load_ps(&m2.data[2 * mres.sz]));
		vw = _mm_mul_ps(vw, _mm_load_ps(&m2.data[3 * mres.sz]));

		// Perform a binary add to reduce cumulative errors
		vx = _mm_add_ps(vx, vz);
		vy = _mm_add_ps(vy, vw);
		vx = _mm_add_ps(vx, vy);

		// Store answer in mres, starting at the index specified
		_mm_store_ps(&mres.data[i*mres.sz], vx);
	}
}

void SIMD_matmul(matd &mres, const matd &m1, const matd &m2)
{
	for (int i = 0; i < mres.sz; i++)
	{
		// Collect the rows of the matrix
		__m256d vx = _mm256_broadcast_sd(&m1.data[i*mres.sz]);
		__m256d vy = _mm256_broadcast_sd(&m1.data[i*mres.sz + 1]);
		__m256d vz = _mm256_broadcast_sd(&m1.data[i*mres.sz + 2]);
		__m256d vw = _mm256_broadcast_sd(&m1.data[i*mres.sz + 3]);
		
		// Perform the multiplication on the rows
		vx = _mm256_mul_pd(vx, _mm256_load_pd(&m2.data[0 * mres.sz]));
		vy = _mm256_mul_pd(vy, _mm256_load_pd(&m2.data[mres.sz]));
		vz = _mm256_mul_pd(vz, _mm256_load_pd(&m2.data[2 * mres.sz]));
		vw = _mm256_mul_pd(vw, _mm256_load_pd(&m2.data[3 * mres.sz]));

		// Perform a binary add to reduce cumulative errors
		vx = _mm256_add_pd(vx, vz);
		vy = _mm256_add_pd(vy, vw);
		vx = _mm256_add_pd(vx, vy);

		// Store answer in mres, starting at the index specified
		_mm256_store_pd(&mres.data[i*mres.sz], vx);
	}
}

//template <typename T>
//void SIMD_matmul(T &mres, const T &m1, const T &m2)
//{
//	for (int i = 0; i < mres.sz; i++)
//	{
//		// Collect the rows of the matrix
//		T vx = _mm256_broadcast_sd(&m1.data[i*mres.sz]);
//		T vy = _mm256_broadcast_sd(&m1.data[i*mres.sz + 1]); // How do you select the correct funtion based on T's type?
//		T vz = _mm256_broadcast_sd(&m1.data[i*mres.sz + 2]);
//		T vw = _mm256_broadcast_sd(&m1.data[i*mres.sz + 3]);
//
//		// Perform the multiplication on the rows
//		vx = _mm256_mul_pd(vx, _mm256_load_pd(&m2.data[0 * mres.sz]));
//		vy = _mm256_mul_pd(vy, _mm256_load_pd(&m2.data[mres.sz]));
//		vz = _mm256_mul_pd(vz, _mm256_load_pd(&m2.data[2 * mres.sz]));
//		vw = _mm256_mul_pd(vw, _mm256_load_pd(&m2.data[3 * mres.sz]));
//
//		// Perform a binary add to reduce cumulative errors
//		vx = _mm256_add_pd(vx, vz);
//		vy = _mm256_add_pd(vy, vw);
//		vx = _mm256_add_pd(vx, vy);
//
//		// Store answer in mres, starting at the index specified
//		_mm256_store_pd(&mres.data[i*mres.sz], vx);
//	}
//}

//void print_mat(const mat &m) {
//  for (int i = 0; i < m.sz; i++) {
//    for (int j = 0; j < m.sz; j++) {
//      std::cout << std::setw(3) << m.data[i*m.sz+j] << ' ';
//    }
//    std::cout << '\n';
//  }
//  std::cout << '\n';
//}
//
//void print_mat(const matd &m) {
//	for (int i = 0; i < m.sz; i++) {
//		for (int j = 0; j < m.sz; j++) {
//			std::cout << std::setw(3) << m.data[i*m.sz + j] << ' ';
//		}
//		std::cout << '\n';
//	}
//	std::cout << '\n';
//}

template <typename T>
void print_mat(const T &m) {
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			std::cout << std::setw(3) << m.data[i*m.sz + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

// A simply initialisation pattern. For a 4x4 matrix:

// 1   2  3  4
// 5   6  7  8
// 9  10 11 12
// 13 14 15 16

template <typename T>
void init_mat(T &m) {
  int count = 1;
  for (int i = 0; i < m.sz; i++) {
    for (int j = 0; j < m.sz; j++) {
      m.data[i*m.sz+j] = count++;
    }
  }
}
//void init_mat(mat &m) {
//  int count = 1;
//  for (int i = 0; i < m.sz; i++) {
//    for (int j = 0; j < m.sz; j++) {
//      m.data[i*m.sz+j] = count++;
//    }
//  }
//}
//
//void init_mat(matd &m) {
//	int count = 1;
//	for (int i = 0; i < m.sz; i++) {
//		for (int j = 0; j < m.sz; j++) {
//			m.data[i*m.sz + j] = count++;
//		}
//	}
//}

// Creates an identity matrix. For a 4x4 matrix:

// 1 0 0 0
// 0 1 0 0
// 0 0 1 0
// 0 0 0 1

//void identity_mat(mat &m) {
//  int count = 0;
//  for (int i = 0; i < m.sz; i++) {
//    for (int j = 0; j < m.sz; j++) {
//      m.data[i*m.sz+j] = (count++ % (m.sz+1)) ? 0 : 1;
//    }
//  }
//}
//
//void identity_mat(matd &m) {
//	int count = 0;
//	for (int i = 0; i < m.sz; i++) {
//		for (int j = 0; j < m.sz; j++) {
//			m.data[i*m.sz + j] = (count++ % (m.sz + 1)) ? 0 : 1;
//		}
//	}
//}

template <typename T>
void identity_mat(T &m) {
	int count = 0;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = (count++ % (m.sz + 1)) ? 0 : 1;
		}
	}
}

int main(int argc, char *argv[])
{
  alignas(sizeof(__m128)) mat mres{new float[SZ*SZ],SZ},mres2{ new float[SZ*SZ],SZ },m{new float[SZ*SZ],SZ},id{new float[SZ*SZ],SZ};
  alignas(sizeof(__m256d)) matd mres3 { new double[SZ*SZ], SZ }, md{ new double[SZ*SZ],SZ }, idd{ new double[SZ*SZ],SZ };
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2, t3, t4, t5, t6;

  std::cout << "Each " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(float)*SZ*SZ << " bytes.\n\n";

  init_mat(m);
  init_mat(md);
  identity_mat(id);
  identity_mat(idd);

  t1 = high_resolution_clock::now();
  matmul(mres,m,id);
  t2 = high_resolution_clock::now();

  std::cout << "/////////////////////////////\n" 
			<< "/// Serial Multiplication ///\n"
			<< "/////////////////////////////\n"
			<< std::endl;

  std::cout << "Initial Matrix" << "\n\n";
  print_mat(m);
  std::cout << "Identity Matrix" << "\n\n";
  print_mat(id);
  std::cout << "Resultant Matrix" << "\n\n";
  print_mat(mres);

  const auto d = duration_cast<microseconds>(t2-t1).count();
  std::cout << "Serial Multiplication took " << d << ' ' << "microseconds.\n\n";

  t3 = high_resolution_clock::now();
  SIMD_matmul(mres2, m, id);
  t4 = high_resolution_clock::now();
  
  std::cout << "//////////////////////////////////////////////\n"
			<< "//// SIMD Single Precision Multiplication ////\n"
			<< "//////////////////////////////////////////////\n"
    		<< std::endl;

  std::cout << "Initial Matrix" << "\n\n";
  print_mat(m);
  std::cout << "Identity Matrix" << "\n\n";
  print_mat(id);
  std::cout << "Resultant Matrix" << "\n\n";
  print_mat(mres2);

  const auto d2 = duration_cast<microseconds>(t4 - t3).count();
  std::cout << "SIMD Single Precision Multiplication took " << d2 << ' ' << "microseconds.\n\n";

  t5 = high_resolution_clock::now();
  SIMD_matmul(mres3, md, idd);
  t6 = high_resolution_clock::now();

  std::cout 
	  << "//////////////////////////////////////////////\n"
	  << "//// SIMD Double Precision Multiplication ////\n"
	  << "//////////////////////////////////////////////\n"
	  << std::endl;

  std::cout << "Initial Matrix" << "\n\n";
  print_mat(md);
  std::cout << "Identity Matrix" << "\n\n";
  print_mat(idd);
  std::cout << "Resultant Matrix" << "\n\n";
  print_mat(mres3);

  const auto d3 = duration_cast<microseconds>(t6 - t5).count();
  std::cout << "SIMD Double Precision Multiplication took " << d3 << ' ' << "microseconds.\n";

  const bool correct = mres == m;
  const bool correct2 = mres2 == m;
  const bool correct3 = mres3 == md;

  delete [] mres.data;
  delete [] mres2.data;
  delete [] m.data;
  delete [] id.data;

#ifdef _WIN32
  system("pause");
#endif

  return correct && correct2 && correct3 ? 0 : -1;
}
