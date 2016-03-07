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

int main(int argc, char ** argv){
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

	dpgmm.updateMeans(X,w,Z);
	std::cout << "Means" << std::endl;
	std::cout << dpgmm.getMeans() << std::endl;

// Test prec update
	std::cout << "Test precisions" << std::endl;
	dpgmm.updatePrecisions(X,w,Z);

	for(int k=0;k<dpgmm.m_Precisions.size();k++){
		std::cout << dpgmm.m_Precisions[k] << std::endl;
	}
	std::cout << dpgmm.m_DOF << std::endl;
	std::cout << dpgmm.m_Scale << std::endl;
	std::cout << dpgmm.m_Bounded_Precisions << std::endl;

//Test concentration update
	std::cout << "Test concentration" << std::endl;
	dpgmm.updateConcentration(Z,w);
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
#if 0
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

	vnl_vector<double> mean(2);
	vnl_vector<double> weights(mesh->GetPoints()->Size());
	for(auto it = mesh->GetPoints()->Begin();it!= mesh->GetPoints()->End();++it){
		MeshType::PointType point =it->Value();
		X(k,0)=point[1];
		X(k,1)=point[2];
		mesh->GetPointData(it.Index(),&weights[k]);
		mean+=X.get_row(k);
		k++;
	}
	mean/=X.rows();

	for(int r=0;r<X.rows();r++){
		X.set_row(r,X.get_row(r)-mean);
	}

	unsigned numberCentroids=3;
	DPGMM dpgmm(numberCentroids,DPGMM::DPGMM_COVARIANCE_SPHERICAL);
	dpgmm.train(X,weights);

	MeshType::Pointer result =MeshType::New();

	vnl_matrix<double> means = dpgmm.getMeans();

	for(unsigned k=0;k<numberCentroids;k++){
		MeshType::PointType point;
		point[0]=0;
		point[1]=means(0,k);
		point[2]=means(1,k);
		result->SetPoint(k,point);
		result->SetPointData(k,1);
	}

	typedef itk::MeshFileWriter<MeshType> MeshFileWriterType;

	MeshFileWriterType::Pointer meshFileWriter = MeshFileWriterType::New();

	meshFileWriter->SetFileName(argv[2]);
	meshFileWriter->SetInput(result);
	meshFileWriter->Update();
#endif
}
