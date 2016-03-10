/*
 * tttDPGMMTest.cpp
 *
 *  Created on: 3 de mar. de 2016
 *      Author: morgan
 */




#include "tttDPGMM.h"

#include <itkMeshFileReader.h>
#include <itkMeshFileWriter.h>
#include <itkMesh.h>

#include "itkImageFileReader.h"
#include "itkImageRegionConstIteratorWithIndex.h"
int main(int argc, char ** argv){
#if 0
	//std::cout << digamma(0.0) << std::endl;
#if 0
	std::cout << digamma(1.0) << std::endl;
	std::cout << digamma(2.0) << std::endl;
	std::cout << digamma(3.0) << std::endl;

	std::cout << gammaln(1.0) << std::endl;
	std::cout << gammaln(2.0) << std::endl;
	std::cout << gammaln(3.0) << std::endl;
#endif

	typedef vnl_matrix<double> MatrixType;
	typedef vnl_vector<double> VectorType;

	MatrixType X(4,2);
	X(0,0)=-1;
	X(0,1)=-1;
	X(1,0)=-1.1;
	X(1,1)=-1.1;
	X(2,0)=1;
	X(2,1)=1;
	X(3,0)=1.1;
	X(3,1)=1.1;

	MatrixType Z(4,2);
	Z.fill(0);

	Z(0,0)=1;
	Z(1,0)=1;
	Z(2,1)=1;
	Z(3,1)=1;

// Test mean update
	DPGMM dpgmm(2,DPGMM::DPGMM_COVARIANCE_SPHERICAL);

	VectorType w(4);
	w.fill(1.0);

	dpgmm.m_Precisions.resize(2);
	dpgmm.m_Precisions[0].set_size(1,1);
	dpgmm.m_Precisions[0].fill(1);

	dpgmm.m_Precisions[1].set_size(1,1);
	dpgmm.m_Precisions[1].fill(1);

	dpgmm.updateMeans(X,Z);
	std::cout << "Means" << std::endl;
	std::cout << dpgmm.getMeans() << std::endl;

// Test prec update
	std::cout << "Test precisions" << std::endl;
	dpgmm.updatePrecisions(X,Z);

	for(int k=0;k<dpgmm.m_Precisions.size();k++){
		std::cout << dpgmm.m_Precisions[k] << std::endl;
	}
	std::cout << dpgmm.m_DOF << std::endl;
	std::cout << dpgmm.m_Scale << std::endl;
	std::cout << dpgmm.m_Bounded_Precisions << std::endl;

//Test concentration update
	std::cout << "Test concentration" << std::endl;
	dpgmm.updateConcentration(Z);
	std::cout << dpgmm.m_Gamma0 << std::endl;
	std::cout << dpgmm.m_Gamma1 << std::endl;
	std::cout << dpgmm.m_Gamma2 << std::endl;


//
	VectorType currentLogprob;
	MatrixType z;
	VectorType bound;
	dpgmm.m_InitialBound=-4.6757541328186907;
	VectorType zeroBound(2);
	zeroBound.fill(0);
	std::cout << "Bound" << std::endl;
	auto p = dpgmm.boundStateLogLikelihood(X,zeroBound,dpgmm.m_Means,dpgmm.m_Precisions);
	std::cout << p << std::endl;


	dpgmm.scoreSamples(X,currentLogprob,z,bound);
	std::cout << "Z" << std::endl;
	std::cout << z << std::endl;
#endif
#if 0
	  typedef double PixelType;
	  const unsigned int Dimension = 2;

	  typedef itk::Image< PixelType, Dimension >       ImageType;

	  typedef itk::ImageFileReader<ImageType> ImageReaderType;
	  ImageReaderType::Pointer imageReader = ImageReaderType::New();
	  imageReader->SetFileName(argv[1]);
	  imageReader->Update();

	  ImageType::Pointer image = imageReader->GetOutput();
	  itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(image,image->GetLargestPossibleRegion());
	  typedef vnl_matrix<double> MatrixType;
	  typedef vnl_vector<double> VectorType;

	  MatrixType X(image->GetLargestPossibleRegion().GetSize(0)*image->GetLargestPossibleRegion().GetSize(1),2);
	  int k=0;
	  VectorType W(image->GetLargestPossibleRegion().GetSize(0)*image->GetLargestPossibleRegion().GetSize(1),2);
	  while(!iterator.IsAtEnd()){
		  ImageType::IndexType index=iterator.GetIndex();
		  ImageType::PointType point;

		  image->TransformIndexToPhysicalPoint(index,point);
		  X(k,0)=point[0];
		  X(k,1)=point[1];
		  W[k]=iterator.Value();
		  ++iterator;
		  k++;
	  }
	  W/=W.sum();
	  W*=W.size();
	  DPGMM dpgmm(100,DPGMM::DPGMM_COVARIANCE_SPHERICAL);
	  dpgmm.train(X,W);


		vnl_vector<double> logprob;
		MatrixType z;
		vnl_vector<double> bound;
		dpgmm.scoreSamples(X,logprob,z,bound);
		typedef itk::Mesh<double,3> MeshType;

		MeshType::Pointer result =MeshType::New();
		 k=0;
		for(unsigned n=0;n<X.rows();n++){
			if(W[n]==0) continue;

			MeshType::PointType point;
			point[0]=0;
			point[1]=X(n,0);
			point[2]=X(n,1);

			result->SetPoint(k,point);
			result->SetPointData(k,z.get_row(n).arg_max());
			k++;
		}
		typedef itk::MeshFileWriter<MeshType> MeshFileWriterType;

		MeshFileWriterType::Pointer meshFileWriter = MeshFileWriterType::New();

		meshFileWriter->SetFileName(argv[2]);
		meshFileWriter->SetInput(result);
		meshFileWriter->Update();
#endif
#if 0
	typedef itk::Mesh<double,3> MeshType;

	typedef vnl_matrix<double> MatrixType;

	MatrixType X(200000,2);
	vnl_random rng;
	for(int k=0;k<100000;k++){
		X(k,0)=rng.normal();
		X(k,1)=10+rng.normal();
	}
	for(int k=100000;k<200000;k++){
		X(k,0)=rng.normal();
		X(k,1)=-10+rng.normal();
	}
	vnl_vector<double> weights(200000);
	unsigned numberCentroids=50;
	DPGMM dpgmm(numberCentroids,DPGMM::DPGMM_COVARIANCE_SPHERICAL);
	weights.fill(1.0);
	dpgmm.train(X,weights);

	vnl_vector<double> logprob;
	MatrixType z;
	vnl_vector<double> bound;
	dpgmm.scoreSamples(X,logprob,z,bound);
	MeshType::Pointer result =MeshType::New();

	for(unsigned n=0;n<X.rows();n++){
		MeshType::PointType point;
		point[0]=0;
		point[1]=X(n,0);
		point[2]=X(n,1);

		result->SetPoint(n,point);
		result->SetPointData(n,z.get_row(n).arg_max());
	}
	typedef itk::MeshFileWriter<MeshType> MeshFileWriterType;

	MeshFileWriterType::Pointer meshFileWriter = MeshFileWriterType::New();

	meshFileWriter->SetFileName(argv[2]);
	meshFileWriter->SetInput(result);
	meshFileWriter->Update();
#endif
#if 1
	typedef itk::Mesh<double,3> MeshType;

	typedef itk::MeshFileReader<MeshType> MeshFileReaderType;

	MeshFileReaderType::Pointer meshFileReader = MeshFileReaderType::New();

	meshFileReader->SetFileName(argv[1]);

	meshFileReader->Update();
	MeshType::Pointer mesh = meshFileReader->GetOutput();
	mesh->DisconnectPipeline();


	typedef vnl_matrix<double> MatrixType;

	MatrixType X(mesh->GetPoints()->Size(),2);

	int k=0;

	vnl_vector<double> sum(2);
	vnl_vector<double> sum2(2);
	sum.fill(0.0);
	sum2.fill(0.0);

	vnl_vector<double> weights(mesh->GetPoints()->Size());
	for(auto it = mesh->GetPoints()->Begin();it!= mesh->GetPoints()->End();++it){
		MeshType::PointType point =it->Value();
		X(k,0)=point[1];
		X(k,1)=point[2];
		mesh->GetPointData(it.Index(),&weights[k]);
		sum+=X.get_row(k);
		sum2+=X.get_row(k).apply(vnl_math_sqr);
		k++;
	}
	auto mean= sum/X.rows();
	auto std=(sum2/X.rows()-mean.apply(vnl_math_sqr)).apply(std::sqrt);

	for(int r=0;r<X.rows();r++){
		X.set_row(r,element_quotient(X.get_row(r)-mean,std));
	}
	//weights.fill(1.0);
	weights=weights/weights.sum();
	weights*=weights.size();
	unsigned numberCentroids=400;
	DPGMM dpgmm(numberCentroids,DPGMM::DPGMM_COVARIANCE_SPHERICAL);
	dpgmm.train(X,weights);

	vnl_vector<double> logprob;
	MatrixType z;
	vnl_vector<double> bound;
	dpgmm.scoreSamples(X,logprob,z,bound);
	MeshType::Pointer result =MeshType::New();

	k=0;
	for(unsigned n=0;n<X.rows();n++){
		MeshType::PointType point;
		point[0]=0;
		point[1]=(std[0]*X(n,0)+mean[0]);
		point[2]=(std[1]*X(n,1)+mean[1]);

		result->SetPoint(k,point);
		result->SetPointData(k,z.get_row(n).arg_max());
		k++;
	}
#if 0
	vnl_matrix<double> means = dpgmm.getMeans();

	for(unsigned k=0;k<numberCentroids;k++){
		MeshType::PointType point;
		point[0]=0;
		point[1]=(std[0]*means(0,k)+mean[0]);
		point[2]=(std[1]*means(1,k)+mean[1]);
		result->SetPoint(k,point);
		result->SetPointData(k,1);
	}
#endif


	typedef itk::MeshFileWriter<MeshType> MeshFileWriterType;

	MeshFileWriterType::Pointer meshFileWriter = MeshFileWriterType::New();

	meshFileWriter->SetFileName(argv[2]);
	meshFileWriter->SetInput(result);
	meshFileWriter->Update();
#endif
}
