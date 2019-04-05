#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <thread>
#ifdef __ORBIS__
#include <kernel.h>
#include <stdlib.h>
size_t sceLibcHeapSize = SCE_LIBC_HEAP_SIZE_EXTENDED_ALLOC_NO_LIMIT;
unsigned int sceLibcHeapExtendedAlloc = 1;
#endif
// $CXX -03 -mavx matmul_assignment.cpp
#if (!defined(_MSC_VER))
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#define SZ (1 << 6)  /*(1 << 10)*/ // == 1024

template<typename M>
struct genericMatrix {
	M *data;
	const size_t sz;
	bool operator==(const genericMatrix &rhs) const {
		return !std::memcmp(data, rhs.data, sz*sz * sizeof(data[0]));
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

#ifdef __ORBIS__
template<typename MatrixType, typename StorageVectorType, typename NumberType,
	typename SetZ, typename L, typename M, typename A, typename S>
	void SIMD_MatMul(MatrixType &mres, const MatrixType &m1, const MatrixType &m2,
		SetZ SetZero, L Load, M Mul, A Add, S Store)
{
	size_t const simdSize = sizeof(StorageVectorType) / sizeof(NumberType);
	StorageVectorType row, column, dotProduct, vsum;
	NumberType columnSections[SZ];
	for (int i = 0; i < mres.sz; i++)
	{
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j];			// 1. Get column data
			}
			vsum = SetZero(0.0f);
			for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
			{
				row = Load(&m1.data[i* mres.sz + z]);				// 2. Get Row Values.
				column = Load(&columnSections[z]);					// 3. Place column values into an __m128
				dotProduct = Mul(row, column);						// 4. Compute dot product of row and column
				vsum = Add(vsum, dotProduct);
			}
			if constexpr (std::is_same<NumberType, float>::value) 
			{
				vsum = _mm_hadd_ps(vsum, vsum);
				vsum = _mm_hadd_ps(vsum, vsum);
			}
			else 
			{
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
			}
			// 5. Reduce to a single float
			mres.data[i*mres.sz + j] = Store(vsum);	            // 6. Store float in appropriate index.
		}
	}
}
template<typename MatrixType, typename StorageType, typename NumberType>
void ThreadedSIMD_MatMul(MatrixType &mres, const MatrixType &m1, const MatrixType &m2, const int startRow, const int rowsToRun)
{
	size_t const simdSize = sizeof(StorageType) / sizeof(NumberType);
	StorageType row, column, dotProduct, vsum;
	NumberType columnSections[SZ];
	for (int i = startRow; i < rowsToRun; i++)
	{
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j];			// 1. Get column data
			}
			if constexpr (std::is_same<NumberType, float>::value)
			{
				vsum = _mm_set_ps1(0.0f);
			}
			else
			{
				vsum = _mm256_set1_pd(0.0f);
			}
			for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
			{
				if constexpr (std::is_same<NumberType, float>::value)
				{
					row = _mm_load_ps(&m1.data[i* mres.sz + z]);
					column = _mm_load_ps(&columnSections[z]);
					dotProduct = _mm_mul_ps(row, column);
					vsum = _mm_add_ps(vsum, dotProduct);
				}
				else
				{
					row = _mm256_load_pd(&m1.data[i* mres.sz + z]);
					column = _mm256_load_pd(&columnSections[z]);
					dotProduct = _mm256_mul_pd(row, column);
					vsum = _mm256_add_pd(vsum, dotProduct);
				}
			}
			if constexpr (std::is_same<NumberType, float>::value) // 5. Reduce to a single 
			{
				vsum = _mm_hadd_ps(vsum, vsum);
				vsum = _mm_hadd_ps(vsum, vsum);
				mres.data[i*mres.sz + j] = _mm_cvtss_f32(vsum);
			}
			else
			{
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				mres.data[i*mres.sz + j] = _mm256_cvtsd_f64(vsum);            // 6. Store float in appropriate index.
			}
		}
	}
}
#else
template<typename StorageVectorType, typename NumberType>
void SIMD_MatMul(const genericMatrix<NumberType> &mres, const genericMatrix<NumberType> &m1, const  genericMatrix<NumberType> &m2)
{
	size_t const simdSize = sizeof(StorageVectorType) / sizeof(NumberType);
	StorageVectorType row, column, dotProduct, vsum;
	NumberType columnSections[SZ];
	for (int i = 0; i < mres.sz; i++)
	{
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j];			// 1. Get column data
			}
			if constexpr (std::is_same<NumberType, float>::value)
			{
				vsum = _mm_set_ps1(0.0f);
			}
			else
			{
				vsum = _mm256_set1_pd(0.0f);
			}
			for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
			{
				if constexpr (std::is_same<NumberType, float>::value)
				{
					row = _mm_load_ps(&m1.data[i* mres.sz + z]);				
					column = _mm_load_ps(&columnSections[z]);			
					dotProduct = _mm_mul_ps(row, column);
					vsum = _mm_add_ps(vsum, dotProduct);
				}
				else
				{
					row = _mm256_load_pd(&m1.data[i* mres.sz + z]);				
					column = _mm256_load_pd(&columnSections[z]);					
					dotProduct = _mm256_mul_pd(row, column);				
					vsum = _mm256_add_pd(vsum, dotProduct);
				}
			}
			if constexpr (std::is_same<NumberType, float>::value) // 5. Reduce to a single 
			{
				vsum = _mm_hadd_ps(vsum, vsum);
				vsum = _mm_hadd_ps(vsum, vsum);
				mres.data[i*mres.sz + j] = _mm_cvtss_f32(vsum);
			}
			else
			{
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				mres.data[i*mres.sz + j] = _mm256_cvtsd_f64(vsum);            // 6. Store float in appropriate index.
			}
		}
	}
}

template<typename StorageVectorType, typename NumberType>
void ThreadedSIMD_MatMul(const genericMatrix<NumberType> &mres, const genericMatrix<NumberType> &m1, const  genericMatrix<NumberType> &m2, const int startRow, const int rowsToProcess)
{
	size_t const simdSize = sizeof(StorageVectorType) / sizeof(NumberType);
	StorageVectorType row, column, dotProduct, vsum;
	NumberType columnSections[SZ];
	for (int i = startRow; i < rowsToProcess; i++)
	{
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j];			// 1. Get column data
			}
			if constexpr (std::is_same<NumberType, float>::value)
			{
				vsum = _mm_set_ps1(0.0f);
			}
			else
			{
				vsum = _mm256_set1_pd(0.0f);
			}
			for (std::size_t z = 0; z < mres.sz; z += simdSize)				// Runs once per vector.	
			{
				if constexpr (std::is_same<NumberType, float>::value)
				{
					row = _mm_load_ps(&m1.data[i* mres.sz + z]);				// 2. Get Row Values.
					column = _mm_load_ps(&columnSections[z]);					// 3. Place column values into an __m128
					dotProduct = _mm_mul_ps(row, column);						// 4. Compute dot product of row and column
					vsum = _mm_add_ps(vsum, dotProduct);
				}
				else
				{
					row = _mm256_load_pd(&m1.data[i* mres.sz + z]);				// 2. Get Row Values.
					column = _mm256_load_pd(&columnSections[z]);					// 3. Place column values into an __m128
					dotProduct = _mm256_mul_pd(row, column);						// 4. Compute dot product of row and column
					vsum = _mm256_add_pd(vsum, dotProduct);
				}
			}
			if constexpr (std::is_same<NumberType, float>::value) // 5. Reduce to a single float
			{
				vsum = _mm_hadd_ps(vsum, vsum);
				vsum = _mm_hadd_ps(vsum, vsum);
				mres.data[i*mres.sz + j] = _mm_cvtss_f32(vsum);
			}
			else 
			{
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum, vsum, 0x20), _mm256_permute2f128_pd(vsum, vsum, 0x31));
				mres.data[i*mres.sz + j] = _mm256_cvtsd_f64(vsum);            // 6. Store float in appropriate index.
			}
		}
	}
}
#endif

// Fastmath /pf:fast flag

int main(int argc, char *argv[])
{
#ifdef __ORBIS__
	const int cpu_mode = sceKernelGetCpumode();

	switch (cpu_mode)
	{
	case SCE_KERNEL_CPUMODE_6CPU: // 0
		std::cout << "6CPU mode \n";
		break;
	case SCE_KERNEL_CPUMODE_7CPU_LOW: // 1
		std::cout << "7CPU (LOW) mode \n";
		break;
	case SCE_KERNEL_CPUMODE_7CPU_NORMAL: // 5
		std::cout << "7CPU ( NORMAL ) mode \n";
		break;
	default:
		std::cout << "CPU broken somehow...\n";
		break;
	}
#endif
	const int num_threads = 6;
	std::thread t[num_threads];

	const unsigned testCaseSize = 6, testCaseIgnoreBuffer = 2;

	alignas(sizeof(__m128))
		genericMatrix<float> mresSerialS{ new float[SZ*SZ],SZ }, mresSIMDS{ new float[SZ*SZ], SZ }, mresThreadedSIMDS{ new float[SZ*SZ], SZ },
		initialMatrixS{ new float[SZ*SZ],SZ }, identityMatrixS{ new float[SZ*SZ],SZ };
	alignas(sizeof(__m256d))
		genericMatrix<double> mresSerialD{ new double[SZ*SZ], SZ }, mresSIMDD{ new double[SZ*SZ], SZ }, mresThreadedSIMDD{ new double[SZ*SZ], SZ },
		initialMatrixD{ new double[SZ*SZ],SZ }, identityMatrixD{ new double[SZ*SZ],SZ };


	using namespace std::chrono;
	using tp_t = time_point<high_resolution_clock>;
	tp_t serialSinglePreTimer, serialSinglePostTimer,
		simdSinglePreTimer, simdSinglePostTimer,
		threadedSIMDSinglePreTimer, threadedSIMDSinglePostTimer,

		serialDoublePreTimer, serialDoublePostTimer,
		simdDoublePreTimer, simdDoublePostTimer,
		threadedSIMDDoublePreTimer, threadedSIMDDoublePostTimer;

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
				<< "Serial SINGLE Precision execution ran in "
				<< std::fixed
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
		<< std::fixed
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
#ifdef  __ORBIS__
		simdSinglePreTimer = high_resolution_clock::now();
		SIMD_MatMul<genericMatrix<float>, __m128, float>
	    (mresSIMDS, initialMatrixS, identityMatrixS, _mm_set_ps1, _mm_load_ps, _mm_mul_ps, _mm_add_ps, _mm_cvtss_f32);
		simdSinglePostTimer = high_resolution_clock::now();
#else
		simdSinglePreTimer = high_resolution_clock::now();
		SIMD_MatMul<__m128, float>(mresSerialS, initialMatrixS, identityMatrixS);
		simdSinglePostTimer = high_resolution_clock::now();
#endif 

		SIMDSingleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(simdSinglePostTimer - simdSinglePreTimer).count();
		if (i >= testCaseIgnoreBuffer)
		{
			std::cout
				<< "SIMD SINGLE Precision execution ran in "
				<< std::fixed
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
		<< std::fixed
		<< std::setprecision(1)
		<< (SIMDExecutionAverage /= testCaseSize)
		<< " microseconds.\n\n";

	const auto singlePrecisionSpeedFactorDifference = (serialSingleExecutionAverage / SIMDExecutionAverage);
	std::cout
		<< "Multiplying a SINGLE Precision Matrix of size " << SZ << 'x' << SZ << ','
		<< "\nSIMD execution was "
		<< std::fixed
		<< std::setprecision(1)
		<< singlePrecisionSpeedFactorDifference
		<< " Times the speed of Serial execution \n" << std::endl;

	double threadedSIMDSingleTimeResult = 0, threadedSIMDSingleExecutionAverage = 0;

	if (SZ >= 64)
	{
		for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
		{
			int threadsPerCore = 0;
			int lastThreadCount = 0;
			int remainder = 0;
#ifdef __ORBIS__
			threadsPerCore = SZ / num_threads;
			remainder = SZ % num_threads;
			int rowExcess = 0;
			for (int i = 0; i < num_threads; i++)
			{
				if (i == num_threads - 1)
				{
					lastThreadCount = threadsPerCore + remainder;
					rowExcess = lastThreadCount;
				}
				else
				{
					rowExcess = threadsPerCore;
				}
				threadedSIMDSinglePreTimer = high_resolution_clock::now();
				t[i] = std::thread(
					ThreadedSIMD_MatMul<genericMatrix<float>,__m128, float>,
					std::ref(mresThreadedSIMDS), std::ref(initialMatrixS), std::ref(identityMatrixS),
					(i * threadsPerCore), (i * threadsPerCore) + rowExcess);

			}
			for (int i = 0; i < num_threads; i++)
			{
				t[i].join();
			}
			threadedSIMDSinglePostTimer = high_resolution_clock::now();
#else
			threadsPerCore = SZ / num_threads;
			remainder = SZ % num_threads;
			int rowExcess = 0;
			for (int i = 0; i < num_threads; i++)
			{
				if (i == num_threads - 1)
				{
					lastThreadCount = threadsPerCore + remainder;
					rowExcess = lastThreadCount;
				}
				else
				{
					rowExcess = threadsPerCore;
				}
				threadedSIMDSinglePreTimer = high_resolution_clock::now();
				t[i] = std::thread(
					ThreadedSIMD_MatMul<__m128, float>,
					std::ref(mresThreadedSIMDS), std::ref(initialMatrixS), std::ref(identityMatrixS),
					(i * threadsPerCore), (i * threadsPerCore) + rowExcess);

			}
			for (int i = 0; i < num_threads; i++)
			{
				t[i].join();
			}
			threadedSIMDSinglePostTimer = high_resolution_clock::now();
#endif
			threadedSIMDSingleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(threadedSIMDSinglePostTimer - threadedSIMDSinglePreTimer).count();
			if (i >= testCaseIgnoreBuffer)
			{
				std::cout
					<< "Threaded SIMD SINGLE precision execution ran in "
					<< std::fixed
					<< std::setprecision(1)
					<< threadedSIMDSingleTimeResult
					<< " microseconds."
					<< std::endl;
				threadedSIMDSingleExecutionAverage += threadedSIMDSingleTimeResult;
			}
		}
		std::cout
			<< "Threaded SIMD SINGLE Precision execution average time after "
			<< testCaseSize
			<< " Iterations was "
			<< std::fixed
			<< std::setprecision(1)
			<< (threadedSIMDSingleExecutionAverage /= testCaseSize)
			<< " microseconds.\n\n";

		const auto singlePrecisionThreadedSpeedFactorDifference = (serialSingleExecutionAverage / threadedSIMDSingleExecutionAverage);
		std::cout
			<< "Multiplying a SINGLE Precision Matrix of size " << SZ << 'x' << SZ << ','
			<< "\nThreaded SIMD execution was "
			<< std::fixed
			<< std::setprecision(1)
			<< singlePrecisionThreadedSpeedFactorDifference
			<< " Times the speed of Serial execution \n" << std::endl;
	}
	else
	{
		std::cout << "Matrix size is less than " << SZ
			<< " Please use a matrix size larger than 64 to engage threading.\n";
	}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Single Precision Serial vs SIMD Execution END /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "Each DOUBLE Precision " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(double)*SZ*SZ << " bytes.\n\n";

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Double Precision Serial vs SIMD Execution END /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "//////////////////////////////////////////////\n"
			<< "/// Serial Double Precision Multiplication ///\n"
			<< "//////////////////////////////////////////////\n"
			<< std::endl;

  double serialDoubleTimeResult = 0, serialDoubleExecutionAverage = 0;
  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
  {
	  serialDoublePreTimer = high_resolution_clock::now();
	  matmul(mresSerialD, initialMatrixD, identityMatrixD);
	  serialDoublePostTimer = high_resolution_clock::now();

	  serialDoubleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(serialDoublePostTimer - serialDoublePreTimer).count();
	  if (i >= testCaseIgnoreBuffer)
	  {
		  std::cout
			  << "Serial DOUBLE Precision execution ran in "
			  << std::fixed
			  << std::setprecision(1)
			  << serialDoubleTimeResult
			  << " microseconds."
			  << std::endl;
		  serialDoubleExecutionAverage += serialDoubleTimeResult;
	  }
  }
  std::cout
	  << "Serial DOUBLE Precision execution average time after "
	  << testCaseSize
	  << " Iterations was "
	  << std::fixed
	  << std::setprecision(1)
	  << (serialDoubleExecutionAverage /= testCaseSize)
	  << " microseconds.\n\n";

  std::cout
	  << "//////////////////////////////////////////////\n"
	  << "//// SIMD Double Precision Multiplication ////\n"
	  << "//////////////////////////////////////////////\n"
	  << std::endl;

  double simdDoubleTimeResult = 0, simdDoubleExecutionAverage = 0;
  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
  {
	
#ifdef __ORBIS__
	  simdDoublePreTimer = high_resolution_clock::now();
	SIMD_MatMul<genericMatrix<double>, __m256d, double>
		(mresSIMDD, initialMatrixD, identityMatrixD,
			_mm256_set1_pd, _mm256_load_pd, _mm256_mul_pd,
			_mm256_add_pd, _mm256_cvtsd_f64);
	simdDoublePostTimer = high_resolution_clock::now();
#else
	simdDoublePreTimer = high_resolution_clock::now();
	SIMD_MatMul<__m256d, double>(mresSIMDD, initialMatrixD, identityMatrixD);
	simdDoublePostTimer = high_resolution_clock::now();
#endif
	simdDoubleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(simdDoublePostTimer - simdDoublePreTimer).count();

	if (i >= testCaseIgnoreBuffer)
	{
		std::cout
			<< "SIMD DOUBLE Precision execution ran in "
			<< std::fixed
			<< std::setprecision(1)
			<< simdDoubleTimeResult
			<< " microseconds."
			<< std::endl;
		simdDoubleExecutionAverage += simdDoubleTimeResult;
	}
  }
  std::cout
	  << "SIMD DOUBLE Precision execution average time after "
	  << testCaseSize
	  << " Iterations was "
	  << std::fixed
	  << std::setprecision(1)
	  << (simdDoubleExecutionAverage /= testCaseSize)
	  << " microseconds.\n\n";

  // Factor by which Parallel execution was faster than Serial execution.
  const auto doublePrecisionSpeedFactorDifference = (serialDoubleExecutionAverage /= simdDoubleExecutionAverage);
  std::cout
	  << "Multiplying a DOUBLE Precision Matrix of size " << SZ << 'x' << SZ << ','
	  << "\nSIMD execution was "
	  << std::fixed
	  << std::setprecision(1)
	  << doublePrecisionSpeedFactorDifference
	  << " Times the speed of Serial execution \n" << std::endl;


  double threadedSIMDDoubleTimeResult = 0, threadedSIMDDoubleExecutionAverage = 0;

  if (SZ >= 64)
  {
	  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
	  {
		  int threadsPerCore = 0;
		  int lastThreadCount = 0;
		  int remainder = 0;
		  #ifdef __ORBIS__
		  threadsPerCore = SZ / num_threads;
		  remainder = SZ % num_threads;
		  int rowExcess = 0;
		  for (int i = 0; i < num_threads; i++)
		  {
			  if (i == num_threads - 1)
			  {
				  lastThreadCount = threadsPerCore + remainder;
				  rowExcess = lastThreadCount;
			  }
			  else
			  {
				  rowExcess = threadsPerCore;
			  }
			  threadedSIMDDoublePreTimer = high_resolution_clock::now();
			  t[i] = std::thread(
				  ThreadedSIMD_MatMul<genericMatrix<double>, __m256d, double>,
				  std::ref(mresThreadedSIMDD), std::ref(initialMatrixD), std::ref(identityMatrixD),
				  (i * threadsPerCore), (i * threadsPerCore) + rowExcess);

		  }
		  for (int i = 0; i < num_threads; i++)
		  {
			  t[i].join();
		  }
		  threadedSIMDDoublePostTimer = high_resolution_clock::now();
	  
	  #else
				threadsPerCore = SZ / num_threads;
				remainder = SZ % num_threads;
				int rowExcess = 0;
				for (int i = 0; i < num_threads; i++)
				{
					if (i == num_threads - 1)
					{
						lastThreadCount = threadsPerCore + remainder;
						rowExcess = lastThreadCount;
					}
					else
					{
						rowExcess = threadsPerCore;
					}
					threadedSIMDDoublePreTimer = high_resolution_clock::now();
					t[i] = std::thread( ThreadedSIMD_MatMul<__m256d, double>,
						std::ref(mresThreadedSIMDD), std::ref(initialMatrixD), std::ref(identityMatrixD),
						(i * threadsPerCore), (i * threadsPerCore) + rowExcess);

				}
				for (int i = 0; i < num_threads; i++)
				{
					t[i].join();
				}
				threadedSIMDDoublePostTimer = high_resolution_clock::now();
	  #endif

		  threadedSIMDDoubleTimeResult = std::chrono::duration<double, std::ratio<1, 1000000>>(threadedSIMDDoublePostTimer - threadedSIMDDoublePreTimer).count();
		  if (i >= testCaseIgnoreBuffer)
		  {
			  std::cout
				  << "Threaded SIMD DOUBLE precision execution ran in "
				  << std::fixed
				  << std::setprecision(1)
				  << threadedSIMDDoubleTimeResult
				  << " microseconds."
				  << std::endl;
			  threadedSIMDDoubleExecutionAverage += threadedSIMDDoubleTimeResult;
		  }
	  }

	  std::cout
		  << "Threaded SIMD DOUBLE Precision execution average time after "
		  << testCaseSize
		  << " Iterations was "
		  << std::fixed
		  << std::setprecision(1)
		  << (threadedSIMDDoubleExecutionAverage /= testCaseSize)
		  << " microseconds.\n\n";

	  const auto doublePrecisionThreadedSpeedFactorDifference = (serialSingleExecutionAverage / threadedSIMDDoubleExecutionAverage);
	  std::cout
		  << "Multiplying a DOUBLE Precision Matrix of size " << SZ << 'x' << SZ << ','
		  << "\nThreaded SIMD DOUBLE execution was "
		  << std::fixed
		  << std::setprecision(1)
		  << doublePrecisionThreadedSpeedFactorDifference
		  << " Times the speed of Serial execution \n" << std::endl;
  }
  else
  {
	  std::cout << "Matrix size is less than " << SZ
		  << " Please use a matrix size larger than 64 to engage threading.\n";
  }


/////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////// Double Precision Serial vs SIMD Execution END /////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

  bool correctSingle = mresSerialS == mresSIMDS;
  bool correctDouble = mresSerialD == mresSIMDD;
  correctSingle = mresSerialS == mresThreadedSIMDS;
  correctDouble = mresSerialD == mresThreadedSIMDD;

  delete [] mresSerialS.data;
  delete [] mresSIMDS.data;
  delete [] mresThreadedSIMDS.data;

  delete [] mresSerialD.data;
  delete [] mresSIMDD.data;
  delete [] mresThreadedSIMDD.data;

  delete [] initialMatrixS.data;
  delete [] initialMatrixD.data;

  delete [] identityMatrixS.data;
  delete [] identityMatrixD.data;

#ifdef _WIN32
  system("pause");
#endif

  return correctSingle && correctDouble ? 0 : -1;
}
