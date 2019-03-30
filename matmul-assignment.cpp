#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
<<<<<<< HEAD
#ifdef __ORBIS__
#include <stdlib.h>
size_t sceLibcHeapSize = SCE_LIBC_HEAP_SIZE_EXTENDED_ALLOC_NO_LIMIT;
unsigned int sceLibcHeapExtendedAlloc = 1;
#endif
=======
#include <vector>
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec

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

<<<<<<< HEAD
template <typename T>
void matmul(T &mres, const T &m1, const T &m2)
=======
void matmul(mat &mres, const mat &m1, const mat &m2)
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec
{
  for (int i = 0; i < mres.sz; i++) {
    for (int j = 0; j < mres.sz; j++) {
      mres.data[i*mres.sz+j] = 0;
      for (int k = 0; k < mres.sz; k++) {
        mres.data[i*mres.sz+j] += m1.data[i*mres.sz+k] * m2.data[k*mres.sz+j];
      }
    }
  }
<<<<<<< HEAD
} 
=======
}
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec

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

<<<<<<< HEAD
void SIMD_MatMul(mat &mres, const mat &m1, const mat &m2) // A way to determine the type of a parameter?
{
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
}


void SIMD_MatMul(matd &mres, const matd &m1, const matd &m2)
{
	size_t const simdSize = sizeof(__m256d) / sizeof(double);
	__m256d row, column, dotProduct, vsum;
	double columnSections[SZ];
	for (int i = 0; i < mres.sz; i++)
	{
=======
float SIMD_VReduce(__m128 v, const size_t size)
{
	float sum;
	__m128 vsum = _mm_set1_ps(0.0f);
	for (std::size_t i = 0; i < size; i += size /* This should be governed by the matrix row/column size later */) {
		vsum = _mm_add_ps(vsum, v);
	}
	//vsum = _mm_load_ps(&v.m128_f32[0]); // Could I use a union here instead?
	vsum = _mm_hadd_ps(vsum, vsum);
	vsum = _mm_hadd_ps(vsum, vsum);
	_mm_store_ss(&sum, vsum);
	return sum;
}

void SIMD_MatMul(mat &mres, const mat &m1, const mat &m2)
{
	__m128 row, column, dotProduct, newRow;
	float reducedFloat;
	std::vector<float> columnSections;
	columnSections.resize(mres.sz);
	std::vector<float> newRowFloats;
	newRowFloats.resize(mres.sz);
	//std::vector<*float> columnSections;
	for (int i = 0; i < mres.sz; i++) 
	{
		row = _mm_load_ps(&m1.data[i*mres.sz]);							// 1. Get Row Values.
		// What intrinsics should be used when the rows are bigger than 4? __m256, etc? Would this need deduced
		// at runtime or simply passed in as a type to be used in the method?
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
<<<<<<< HEAD
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
=======
				columnSections[y] = m2.data[y *mres.sz + j];			// 2. Get column data

				// Use add_ps here instead to load via index?
			}
		
			column = _mm_load_ps(&columnSections[0]); // 3. Place column values into an __m128
			// Any difference between using a pointer on the array / reference to load and not here?);
			//resultantRow[j] = _mm_mul_ps(_mm_load_ps(&m1.data[j*mres.sz]), column[i]);
			/*_mm_dp_ps(row, column, What is the third parameter here?);
			https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_dp_ps&expand=2160
			*/
			dotProduct = _mm_mul_ps(row, column);						// 4. Compute dot product of row and column
			reducedFloat = SIMD_VReduce(dotProduct, mres.sz);			// 5. Reduce to a single float
			newRowFloats[j] = reducedFloat;								 // 6. Store float in appropriate index.
		} // Repeat for every column
		newRow = _mm_load_ps(&newRowFloats[0]);
		_mm_store_ps(&mres.data[i*mres.sz], newRow);					// 7. Add new row into Matrix 
	} // Repeat for every row
}

//double SIMD_VReduce(__m256d v, const size_t size)
//{
//	double sum;
//	__m256d vsum = _mm256_set1_pd(0.0f);
//	for (std::size_t i = 0; i < size; i += size) {
//		vsum = _mm256_add_pd(vsum, v);
//	}
//	vsum = _mm256_hadd_pd(vsum, vsum);
//	vsum = _mm256_hadd_pd(vsum, vsum);
//	_mm_store_sd(&sum, vsum);
//	/* Not sure this is right. _mm256_store_pd will store the whole intrinsic, but _mm_store_ss will Store the lowest 32 bit float of a into memory.
//	 What is the double precision equivilent?
//	 */
//	return sum;
//}
//void SIMD_MatMul(matd &mres, const matd &m1, const matd &m2, const int matrix_dimension)
//{
//	__m256d row, column, dotProduct, newRow;
//	double reducedDouble;
//	std::vector<double> columnSections;
//	columnSections.resize(mres.sz);
//	std::vector<double> newRowDoubles;
//	newRowDoubles.resize(mres.sz);
//	for (int i = 0; i < mres.sz; i++)
//	{
//		row = _mm256_load_pd(&m1.data[i*mres.sz]); // 1. Get Row Values.
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec
//		for (int j = 0; j < mres.sz; j++)
//		{
//			for (int y = 0; y < mres.sz; y++)
//			{
<<<<<<< HEAD
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
=======
//				columnSections[y] = m2.data[y *mres.sz + j]; // 2. Get column data
//			}
//			column = _mm256_load_pd(&columnSections[0]); // 3. Place column values into an __m128
//			dotProduct = _mm256_mul_pd(row, column); // 4. Compute dot product of row and column
//			reducedDouble = SIMD_VReduce(dotProduct, matrix_dimension); // 5. Reduce to a single float
//			newRowDoubles[j] = reducedDouble; // 6. Store float in appropriate index.
//		}
//		newRow = _mm256_load_pd(&newRowDoubles[0]);
//		_mm256_store_pd(&mres.data[i*mres.sz], newRow); // 7. Add new row into Matrix 
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec
//	}
//}

// Fastmath /pf:fast flag
// Look for custom memory allocator in slides to align memeory appropriately

int main(int argc, char *argv[])
{
<<<<<<< HEAD
	const unsigned testCaseSize = 5, testCaseIgnoreBuffer = 2;
=======
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec
  //std::size_t size = SZ * SZ * sizeof(float);
  //std::size_t space = size + 16;
  //void *p = std::malloc(space);
  
  //void *pp = std::align(16, size, p, space);
  //std::aligned_alloc(16, SZ*SZ*sizeof(float));
<<<<<<< HEAD
  alignas(sizeof(__m128)) mat mresSerialS{ new float[SZ*SZ],SZ}, mresSIMDS{ new float[SZ*SZ],SZ }, mresSIMDS2{ new float[SZ*SZ],SZ }, initialMatrixS{new float[SZ*SZ],SZ}, identityMatrixS{new float[SZ*SZ],SZ};
  //alignas(sizeof(__m128d)) matd mresSerialD { new double[SZ*SZ], SZ }, mresSIMDD { new double[SZ*SZ], SZ }, initialMatrixD{ new double[SZ*SZ],SZ }, identityMatrixD{ new double[SZ*SZ],SZ };
  alignas(sizeof(__m256d)) matd mresSerialD { new double[SZ*SZ], SZ }, mresSIMDD{ new double[SZ*SZ], SZ }, initialMatrixD{ new double[SZ*SZ],SZ }, identityMatrixD{ new double[SZ*SZ],SZ };
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t serialSinglePreTimer, serialSinglePostTimer,
	  simdSinglePreTimer, simdSinglePostTimer,
=======
  alignas(sizeof(__m128)) mat mres{ new float[SZ*SZ],SZ},mres2{ new float[SZ*SZ],SZ },m{new float[SZ*SZ],SZ}, id{new float[SZ*SZ],SZ};
  alignas(sizeof(__m256d)) matd mres3 { new double[SZ*SZ], SZ }, md{ new double[SZ*SZ],SZ }, idd{ new double[SZ*SZ],SZ };
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2, t7, t8, t9, t10;
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec

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

<<<<<<< HEAD
  std::cout << "Each DOUBLE Precision " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(double)*SZ*SZ << " bytes.\n\n";
=======
  t7 = high_resolution_clock::now();
  SIMD_MatMul(mres2, m, id);
  t8 = high_resolution_clock::now();

  /*
	m X m (4x4) should be 

	//  90 100 110 120
	// 202 228 254 280
	// 314 356 398 440
	// 426 484 542 600
  
  */

  std::cout
	  << "//////////////////////////////////////////////\n"
	  << "//// SIMD Single Precision Multiplication ////\n"
	  << "//////////////////////////////////////////////\n"
	  << std::endl;
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec

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

<<<<<<< HEAD
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
=======
  const auto d4 = duration_cast<microseconds>(t8 - t7).count();
  std::cout << "SIMD Single Precision Multiplication took " << d4 << ' ' << "microseconds.\n";

  //t9 = high_resolution_clock::now();
  //SIMD_MatMul(mres3, md, idd, SZ); // It's right but only by accident
  //t10 = high_resolution_clock::now();

  //std::cout
	 // << "//////////////////////////////////////////////\n"
	 // << "//// SIMD Double Precision Multiplication ////\n"
	 // << "//////////////////////////////////////////////\n"
	 // << std::endl;

  //std::cout << "Initial Matrix" << "\n\n";
  //print_mat(md);
  //std::cout << "Identity Matrix" << "\n\n";
  //print_mat(idd);
  //std::cout << "Resultant Matrix" << "\n\n";
  //print_mat(mres3);

  //const auto d5 = duration_cast<microseconds>(t10 - t9).count();
  //std::cout << "SIMD Double Precision Multiplication took " << d5 << ' ' << "microseconds.\n";

  const bool correct = mres == m;
  const bool correct2 = mres2 == m;
 // const bool correct3 = mres3 == md;

 // const bool correct3 = mres3 == md;
 // const bool correct4 = mres4 == m;

  delete [] mres.data;
  delete [] mres2.data;
  delete [] mres3.data;
  delete [] m.data;
  delete [] md.data;
  delete [] id.data;
  delete [] idd.data;
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec

#ifdef _WIN32
  system("pause");
#endif

<<<<<<< HEAD
  return correctSingle && correctDouble /*&& correctDouble256*/ ? 0 : -1;
=======
  return correct && correct2 ? 0 : -1;
>>>>>>> 17d2782486a33924211ee43fa37f8988fcdb5dec
}
