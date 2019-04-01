#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <thread>
#include <memory>

#ifdef __ORBIS__
#include <stdlib.h>
size_t sceLibcHeapSize = SCE_LIBC_HEAP_SIZE_EXTENDED_ALLOC_NO_LIMIT;
// Add to include path: %SCE_ORBIS_SDK_DIR%/target/include
#include <kernel.h>
unsigned int sceLibcHeapExtendedAlloc = 1;
#else
//#include <pthread.h>
//#define ScePthread            pthread_t
//#define scePthreadCreate      pthread_create
//#define scePthreadJoin        pthread_join
//#define ScePthreadMutex       pthread_mutex_t
//#define scePthreadMutexLock   pthread_mutex_lock
//#define scePthreadMutexUnlock pthread_mutex_unlock
#endif

// $CXX -03 -mavx matmul_assignment.cpp

#if (!defined(_MSC_VER))
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#define SZ (1 << 2)  /*(1 << 10)*/ // == 1024
// use new align
// use std::begin and std::end on all containter now

struct mat {
  float *data;
  const size_t sz;
  bool operator==(const mat &rhs) const {
    return !std::memcmp(data,rhs.data,sz*sz*sizeof(data[0]));
  }
};
 // Can I make this generic by passing in the float / double type as a parameter?
struct matd {
	double *data;
	const size_t sz;
	bool operator==(const matd &rhs) const {
		return !std::memcmp(data, rhs.data,sz*sz * sizeof(data[0]));
	}
};

template <typename T>
void matmul(T &mres, const T &m1, const T &m2)
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

template <typename T>
void print_mat(const T &m) {
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			std::cout << std::setw(3) << std::setprecision(0) << m.data[i*m.sz + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

// A simple initialisation pattern. For a 4x4 matrix:

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

// Creates an identity matrix. For a 4x4 matrix:

// 1 0 0 0
// 0 1 0 0
// 0 0 1 0
// 0 0 0 1

template <typename T>
void identity_mat(T &m) {
	int count = 0;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = (count++ % (m.sz + 1)) ? 0 : 1;
		}
	}
}

void SIMD_MatMul(mat &mres, const mat &m1, const mat &m2) // A way to determine the type of a parameter?
{
	pthread_t thread[num_threads];
	using tup_t = decltype (std::make_tuple(&mres.data[0], &m1.data[0], &m2.data[0], SZ));
	tup_t ptrs[num_threads];

	auto tproc = [](void * arg) -> void * 
	{
		auto &[mres, m1, m2, sz] = *static_cast <tup_t *>(arg);

		size_t const simdSize = sizeof(__m128) / sizeof(float);
		__m128 row, column, dotProduct, vsum;
		float columnSections[SZ];
		for (int i = 0; i < mres.sz; i++)
		{
			for (int j = 0; j < mres.sz; j++)
			{
				for (int y = 0; y < mres.sz; y++)
				{
					columnSections[y] = m2.data[y *mres.sz + j];			// 1. Get column data
				}
				/*_mm_dp_ps(row, column, What is the third parameter here?);
				https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_dp_ps&expand=2160
				*/
				vsum = _mm_set_ps1(0.0f);
				for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
				{
					row = _mm_load_ps(&m1.data[i* mres.sz + z]);				// 2. Get Row Values.
					column = _mm_load_ps(&columnSections[z]);					// 3. Place column values into an __m128
					dotProduct = _mm_mul_ps(row, column);						// 4. Compute dot product of row and column
					vsum = _mm_add_ps(vsum, dotProduct);
				}
				vsum = _mm_hadd_ps(vsum, vsum);
				vsum = _mm_hadd_ps(vsum, vsum);								// 5. Reduce to a single float
				mres.data[i*mres.sz + j] = _mm_cvtss_f32(vsum);
				//_mm_store_ss(&mres.data[i*mres.sz + j], vsum);	            // 6. Store float in appropriate index.
			}
		}
		return nullptr;
	}

	const auto chunk = SZ / num_threads;
	for (unsigned i = 0; i < num_threads; i++) 
	{
		ptrs[i] = std::make_tuple(&mres.data[i * chunk], &m1.data[i * chunk], &m2.data[i * chunk], chunk);
		pthread_create(&thread[i], nullptr, tproc, &ptrs[i]);
	}

	for (const auto &t : thread) { pthread_join(t, nullptr); }

}

void SIMD_MatMul(matd &mres, const matd &m1, const matd &m2)
{
	size_t const simdSize = sizeof(__m256d) / sizeof(double);
	__m256d row, column, dotProduct, vsum;
	double columnSections[SZ];
	for (int i = 0; i < mres.sz; i++)
	{
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j];				// 1. Get column data
			}
			vsum = _mm256_set1_pd(0.0);
			for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
			{
				row = _mm256_load_pd(&m1.data[i* mres.sz + z]);				// 2. Get Row Values.
				column = _mm256_load_pd(&columnSections[z]);				// 3. Place column values into an __m256d
				dotProduct = _mm256_mul_pd(row, column);					// 4. Compute dot product of row and column
				vsum = _mm256_add_pd(vsum, dotProduct);
			}
			vsum = _mm256_hadd_pd(vsum, vsum); // This seems to go wrong - vsums the first 2 instead of 1,3 -> 2,4. Is this alignment? Consider broadcast here instead.
			vsum = _mm256_hadd_pd(vsum, vsum); // 5. Reduce to a single float
			mres.data[i*mres.sz + j] = _mm256_cvtsd_f64(vsum);	            // 6. Store float in appropriate index.
		}
	}
}

//void SIMD_MatMul(matd &mres, const matd &m1, const matd &m2)
//{
//	size_t const simdSize = sizeof(__m128d) / sizeof(double);
//	__m128d row, column, dotProduct, vsum;
//	double columnSections[SZ];
//	for (int i = 0; i < mres.sz; i++)
//	{
//		for (int j = 0; j < mres.sz; j++)
//		{
//			for (int y = 0; y < mres.sz; y++)
//			{
//				columnSections[y] = m2.data[y *mres.sz + j];			// 1. Get column data
//			}
//			vsum = _mm_set1_pd(0.0);
//			for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
//			{
//				row = _mm_load_pd(&m1.data[i* mres.sz + z]);				// 2. Get Row Values.
//				column = _mm_load_pd(&columnSections[z]);					// 3. Place column values into an __m128
//				dotProduct = _mm_mul_pd(row, column);						// 4. Compute dot product of row and column
//				vsum = _mm_add_pd(vsum, dotProduct);
//			}
//			vsum = _mm_hadd_pd(vsum, vsum);	
//			// 5. Reduce to a single float
//			_mm_store_sd(&mres.data[i*mres.sz + j], vsum);	            // 6. Store float in appropriate index.
//		}
//	}
//}

// Fastmath /pf:fast flag
// Look for custom memory allocator in slides to align memeory appropriately

int main(int argc, char *argv[])
{
	const unsigned testCaseSize = 5, testCaseIgnoreBuffer = 2, num_threads = 8;
  //std::size_t size = SZ * SZ * sizeof(float);
  //std::size_t space = size + 16;
  //void *p = std::malloc(space);
  
  //void *pp = std::align(16, size, p, space);
  //std::aligned_alloc(16, SZ*SZ*sizeof(float));
  alignas(sizeof(__m128)) mat mresSerialS{ new float[SZ*SZ],SZ}, mresSIMDS{ new float[SZ*SZ],SZ }, mresSIMDS2{ new float[SZ*SZ],SZ }, initialMatrixS{new float[SZ*SZ],SZ}, identityMatrixS{new float[SZ*SZ],SZ};
  //alignas(sizeof(__m128d)) matd mresSerialD { new double[SZ*SZ], SZ }, mresSIMDD { new double[SZ*SZ], SZ }, initialMatrixD{ new double[SZ*SZ],SZ }, identityMatrixD{ new double[SZ*SZ],SZ };
  alignas(sizeof(__m256d)) matd mresSerialD { new double[SZ*SZ], SZ }, mresSIMDD{ new double[SZ*SZ], SZ }, initialMatrixD{ new double[SZ*SZ],SZ }, identityMatrixD{ new double[SZ*SZ],SZ };
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t serialSinglePreTimer, serialSinglePostTimer,
	  simdSinglePreTimer, simdSinglePostTimer,

	  serialDoublePreTimer, serialDoublePostTimer,
	  simdDoublePreTimer, simdDoublePostTimer;

  init_mat(initialMatrixS);
  init_mat(initialMatrixD);
  identity_mat(identityMatrixS);
  identity_mat(identityMatrixD);

  std::cout << "Each SINGLE Precision " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(float)*SZ*SZ << " bytes.\n\n";

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Single Precision Serial vs SIMD Execution BEGIN ///////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "//////////////////////////////////////////////\n"
			<< "/// Serial Single Precision Multiplication ///\n"
			<< "//////////////////////////////////////////////\n"
			<< std::endl;
  double serialSingleTimeResult = 0, serialSingleExecutionAverage = 0;
  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
  {
	  serialSinglePreTimer = high_resolution_clock::now();
	  matmul(mresSerialS, initialMatrixS, identityMatrixS);
	  serialSinglePostTimer = high_resolution_clock::now();

	  serialSingleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(serialSinglePostTimer - serialSinglePreTimer).count();
	  if (i >= testCaseIgnoreBuffer)
	  {
		  std::cout
			  << "Serial SINGE Precision execution ran in "
			  << std::setprecision(1)
			  << serialSingleTimeResult
			  << " microseconds."
			  << std::endl;
		  serialSingleExecutionAverage += serialSingleTimeResult;
	  }
  }
  std::cout
	  << "Serial SINGLE Precision execution average time after "
	  << testCaseSize
	  << " Iterations was "
	  << std::setprecision(1)
	  << (serialSingleExecutionAverage /= testCaseSize)
	  << " microseconds.\n\n";

  std::cout
	  << "//////////////////////////////////////////////\n"
	  << "//// SIMD Single Precision Multiplication ////\n"
	  << "//////////////////////////////////////////////\n"
	  << std::endl;

  double SIMDSingleTimeResult = 0, SIMDExecutionAverage = 0;
  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
  {
	  simdSinglePreTimer = high_resolution_clock::now();
	  SIMD_MatMul(mresSIMDS, initialMatrixS, identityMatrixS);
	  simdSinglePostTimer = high_resolution_clock::now();

	  SIMDSingleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(simdSinglePostTimer - simdSinglePreTimer).count();
	  if (i >= testCaseIgnoreBuffer)
	  {
		  std::cout
			  << "SIMD SINGLE precision execution ran in "
			  << std::setprecision(1)
			  << SIMDSingleTimeResult
			  << " microseconds."
			  << std::endl;
		  SIMDExecutionAverage += SIMDSingleTimeResult;
	  }
  }

  std::cout
	  << "SIMD SINGLE Precision execution average time after "
	  << testCaseSize
	  << " Iterations was "
	  << std::setprecision(1)
	  << (SIMDExecutionAverage /= testCaseSize)
	  << " microseconds.\n\n";

  // Factor by which Parallel execution was faster than Serial execution.
  const auto singlePrecisionSpeedFactorDifference = (serialSingleExecutionAverage /= SIMDExecutionAverage);
  std::cout
	  << "Multiplying a SINGLE Precision Matrix of size " << SZ << 'x' << SZ << ','
	  << "\nSIMD execution was "
	  << std::setprecision(1)
	  << singlePrecisionSpeedFactorDifference
	  << " Times the speed of Serial execution \n" << std::endl;

  std::cout << "Initial Matrix" << "\n\n";
  print_mat(initialMatrixS);
  std::cout << "Identity Matrix" << "\n\n";
  print_mat(identityMatrixS);
  std::cout << "Resultant Matrix" << "\n\n";
  print_mat(mresSIMDS);

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Single Precision Serial vs SIMD Execution END /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "Each DOUBLE Precision " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(double)*SZ*SZ << " bytes.\n\n";

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Double Precision Serial vs SIMD Execution END /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  serialDoublePreTimer = high_resolution_clock::now();
  matmul(mresSerialD, initialMatrixD, identityMatrixD);
  serialDoublePostTimer = high_resolution_clock::now();

  std::cout << "//////////////////////////////////////////////\n"
	        << "/// Serial Double Precision Multiplication ///\n"
	        << "//////////////////////////////////////////////\n"
	        << std::endl;

  const auto serialDoubleTime = duration_cast<microseconds>(serialDoublePostTimer - serialDoublePreTimer).count();
  std::cout << "Serial Multiplication took " << serialDoubleTime << ' ' << "microseconds.\n\n";

  simdDoublePreTimer = high_resolution_clock::now();
  SIMD_MatMul(mresSIMDD, initialMatrixD, identityMatrixD);
  simdDoublePostTimer = high_resolution_clock::now();

  std::cout
	  << "//////////////////////////////////////////////\n"
	  << "//// SIMD Double Precision Multiplication ////\n"
	  << "//////////////////////////////////////////////\n"
	  << std::endl;

  std::cout << "Initial Matrix" << "\n\n";
  print_mat(initialMatrixD);
  std::cout << "Identity Matrix" << "\n\n";
  print_mat(identityMatrixD);
  std::cout << "Resultant Matrix" << "\n\n";
  print_mat(mresSIMDD);

  const auto d5 = duration_cast<microseconds>(simdDoublePostTimer - simdDoublePreTimer).count();
  std::cout << "SIMD Double Precision Multiplication took " << d5 << ' ' << "microseconds.\n";

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Double Precision Serial vs SIMD Execution END /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  const bool correctSingle = mresSerialS == mresSIMDS; // Compare Serial single precision result vs simd single precision result
  const bool correctDouble = mresSerialD == mresSIMDD; // Compare Serial double precision result vs simd double precision result

  delete [] mresSerialS.data;
  delete [] mresSIMDS.data;

  delete [] mresSerialD.data;
  delete [] mresSIMDD.data;

  delete [] initialMatrixS.data;
  delete [] initialMatrixD.data;

  delete [] identityMatrixS.data;
  delete [] identityMatrixD.data;

#ifdef _WIN32
  system("pause");
#endif

  return correctSingle && correctDouble /*&& correctDouble256*/ ? 0 : -1;
}
