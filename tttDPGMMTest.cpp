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

}
