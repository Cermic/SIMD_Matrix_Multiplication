#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <vector>

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

struct matd {
	double *data;
	const size_t sz;
	bool operator==(const matd &rhs) const {
		return !std::memcmp(data, rhs.data,sz*sz * sizeof(data[0]));
	}
};

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

double SIMD_VReduce(__m256d v, const size_t size)
{
	double sum;
	__m256d vsum = _mm256_set1_pd(0.0f);
	for (std::size_t i = 0; i < size; i += size) {
		vsum = _mm256_add_pd(vsum, v);
	}
	vsum = _mm256_hadd_pd(vsum, vsum);
	vsum = _mm256_hadd_pd(vsum, vsum);
	_mm256_store_sd(&sum, vsum);
	return sum;
}

void SIMD_MatMul(mat &mres, const mat &m1, const mat &m2, const int matrix_dimension)
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
		row = _mm_load_ps(&m1.data[i*mres.sz]); // 1. Get Row Values.
		// What intrinsics should be used when the rows are bigger than 4? __m256, etc? Would this need deduced
		// at runtime or simply passed in as a type to be used in the method?
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j]; // 2. Get column data

				// Use add_ps here instead to load via index?
			}
		
			column = _mm_load_ps(&columnSections[0]); // 3. Place column values into an __m128
			// Any difference between using a pointer on the array / reference to load and not here?);
			//resultantRow[j] = _mm_mul_ps(_mm_load_ps(&m1.data[j*mres.sz]), column[i]);
			/*_mm_dp_ps(row, column, What is the third parameter here?);
			https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_dp_ps&expand=2160
			*/
			dotProduct = _mm_mul_ps(row, column); // 4. Compute dot product of row and column
			reducedFloat = SIMD_VReduce(dotProduct, matrix_dimension); // 5. Reduce to a single float
			newRowFloats[j] = reducedFloat; // 6. Store float in appropriate index.
		}
		newRow = _mm_load_ps(&newRowFloats[0]);
		_mm_store_ps(&mres.data[i*mres.sz], newRow); // 7. Add new row into Matrix 
	}
}

void SIMD_MatMul(matd &mres, const matd &m1, const matd &m2, const int matrix_dimension)
{
	__m256d row, column, dotProduct, newRow;
	double reducedDouble;
	std::vector<double> columnSections;
	columnSections.resize(mres.sz);
	std::vector<double> newRowDoubles;
	newRowDoubles.resize(mres.sz);
	for (int i = 0; i < mres.sz; i++)
	{
		row = _mm256_load_pd(&m1.data[i*mres.sz]); // 1. Get Row Values.
		for (int j = 0; j < mres.sz; j++)
		{
			for (int y = 0; y < mres.sz; y++)
			{
				columnSections[y] = m2.data[y *mres.sz + j]; // 2. Get column data
			}
			column = _mm256_load_pd(&columnSections[0]); // 3. Place column values into an __m128
			dotProduct = _mm256_mul_pd(row, column); // 4. Compute dot product of row and column
			reducedDouble = SIMD_VReduce(dotProduct, matrix_dimension); // 5. Reduce to a single float
			newRowDoubles[j] = reducedDouble; // 6. Store float in appropriate index.
		}
		newRow = _mm256_load_pd(&newRowDoubles[0]);
		_mm256_store_pd(&mres.data[i*mres.sz], newRow); // 7. Add new row into Matrix 
	}
}

// Fastmath /pf:fast flag
// Look for custom memory allocator in slides to align memeory appropriately

int main(int argc, char *argv[])
{
  std::size_t size = SZ * SZ * sizeof(float);
  std::size_t space = size + 16;
  void *p = std::malloc(space);
  
  void *pp = std::align(16, size, p, space);
  //std::aligned_alloc(16, SZ*SZ*sizeof(float));
  alignas(sizeof(__m128)) mat mres{ new float[SZ*SZ],SZ},mres2{ new float[SZ*SZ],SZ },m{new float[SZ*SZ],SZ}, mres4{ new float[SZ*SZ], SZ }, id{new float[SZ*SZ],SZ};
  alignas(sizeof(__m256d)) matd mres3 { new double[SZ*SZ], SZ }, /*mres4{ new double[SZ*SZ], SZ },*/ md{ new double[SZ*SZ],SZ }, idd{ new double[SZ*SZ],SZ };
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2, t3, t4, t5, t6, t7, t8;

  std::cout << "Each " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(float)*SZ*SZ << " bytes.\n\n";

  init_mat(m);
  //init_mat(md);
  identity_mat(id);
  //identity_mat(idd);

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

  t7 = high_resolution_clock::now();
  SIMD_MatMul(mres2, m, id, SZ); // It's right but only by accident
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

  std::cout << "Initial Matrix" << "\n\n";
  print_mat(m);
  std::cout << "Identity Matrix" << "\n\n";
  print_mat(id);
  std::cout << "Resultant Matrix" << "\n\n";
  print_mat(mres2);

  const auto d4 = duration_cast<microseconds>(t8 - t7).count();
  std::cout << "SIMD Single Precision Multiplication took " << d4 << ' ' << "microseconds.\n";

  const bool correct = mres == m;
  const bool correct2 = mres2 == m;
 // const bool correct3 = mres3 == md;
 // const bool correct4 = mres4 == m;

  delete [] mres.data;
  delete [] mres2.data;
  //delete [] mres3.data;
  //delete [] mres4.data;
  delete [] m.data;
  //delete [] md.data;
  delete [] id.data;
 // delete [] idd.data;

#ifdef _WIN32
  system("pause");
#endif

  return correct && correct2 ? 0 : -1;
}
