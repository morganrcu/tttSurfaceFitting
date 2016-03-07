
#include <itkImageToMeshFilter.h>
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkSymmetricEigenAnalysis.h>
#include <itkGaussianMembershipFunction.h>
#include <itkAutomaticTopologyMeshSource.h>
#include <itkMeshFileWriter.h>
#include <itkMesh.h>
#include <itkHistogram.h>
#include <itkOtsuThresholdCalculator.h>
#include <itkRenyiEntropyThresholdCalculator.h>
#include <itkLiThresholdCalculator.h>
#include <itkIsoDataThresholdCalculator.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <vnl/algo/vnl_cholesky.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/vnl_matlab_filewrite.h>

#include <fstream>
namespace ttt {
template<class TInputImage,class TOutputMesh> class SphereFittingCalculator: public itk::ImageToMeshFilter<TInputImage,TOutputMesh> {
public:
	typedef SphereFittingCalculator Self;
	typedef itk::Object Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self> ConstPointer;

	itkNewMacro(Self)
	;itkTypeMacro(SphereFittingCalculator,itk::Object)
	;

	typedef TInputImage InputImageType;
	typedef typename TInputImage::ConstPointer ImageConstPointer;
	typedef typename InputImageType::PointType PointType;

	typedef typename PointType::VectorType VectorType;

	typedef typename itk::SymmetricSecondRankTensor<typename PointType::CoordRepType,PointType::Dimension> SymmetricMatrixType;
	typedef typename itk::Matrix<typename PointType::CoordRepType,PointType::Dimension,PointType::Dimension> MatrixType;


	SymmetricMatrixType outerProduct(const VectorType & a, const VectorType & b){
		SymmetricMatrixType matrix;
		for(int r=0;r<PointType::Dimension;r++){
			for(int c=0;c<PointType::Dimension;c++){
				matrix(r,c)=a[r]*b[c];
			}
		}
		return matrix;
	}

	void ComputeCenterOfMass() {
		//1. Compute center of mass
		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(), this->GetInput()->GetLargestPossibleRegion());

		PointType origin = this->GetInput()->GetOrigin();
		VectorType sum;
		sum.Fill(0);
		m_TotalIntensity=0;
		m_Count=0;
		while (!iterator.IsAtEnd()) {

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(), point);

			VectorType vector = (point - origin);
			double weight = iterator.Get();
			if(weight<0) weight=0;
			sum += weight * vector;
			//std::cout << weight << "\t" << vector << "\t" << sum << std::endl;
			m_TotalIntensity += weight;
			++iterator;
			m_Count++;
		}
		sum = sum / ( m_TotalIntensity);

		m_CenterOfMass = origin + sum;
		std::cout << m_CenterOfMass << std::endl;
	}

	void ComputeSphereFitting(){

		m_Center=m_CenterOfMass;

		for(int it=0;it<2000;it++){

			typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

			IteratorType iterator(this->GetInput(), this->GetInput()->GetLargestPossibleRegion());
			m_Radius=0;
			VectorType L;
			L.Fill(0);
			while (!iterator.IsAtEnd()) {
				PointType point;
				this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(), point);

				VectorType vector = ( point-m_Center);
				double weight = iterator.Get();
				if(weight<0) weight=0;
				auto Li=vector.GetNorm();

				m_Radius+=weight*Li;
				L+=weight*-vector/Li;

				++iterator;

			}
			m_Radius=m_Radius/(m_TotalIntensity);
			L=L/(m_TotalIntensity);
			m_Center= m_CenterOfMass + L*m_Radius;


			iterator.GoToBegin();
			double f=0;
			while(!iterator.IsAtEnd()){
				PointType point;
				this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(), point);

				PointType projection = m_Center + (point - m_Center)*m_Radius;

				double weight = iterator.Get();
				if(weight<0) weight=0;

				f+=weight*std::pow((projection-point).GetNorm()-m_Radius,2);
				++iterator;
			}

			std::cout <<"it " << it << "\t" << m_Center << " " << m_Radius << "\t" << f << std::endl;
		}

	}

	void ComputeProjectionCovariance(){

		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(),this->GetInput()->GetLargestPossibleRegion());

		m_ProjectionCovariance.Fill(0);
		while(!iterator.IsAtEnd()){

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),point);

			PointType projection = m_Center + (point - m_Center)*m_Radius;

			m_ProjectionCovariance+=this->outerProduct(point-projection,point-projection)*iterator.Get();

			++iterator;
		}
		m_ProjectionCovariance/=(m_TotalIntensity);
	}
	void GenerateData() {

		this->ComputeCenterOfMass();
		this->ComputeSphereFitting();

		this->ComputeProjectionCovariance();

		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(), this->GetInput()->GetLargestPossibleRegion());



		MatrixType covarianceMatrix;
#if 0
		covarianceMatrix[0][0]=m_ProjectionCovariance(0,0);
		covarianceMatrix[0][1]=m_ProjectionCovariance(1,0);
		covarianceMatrix[0][2]=m_ProjectionCovariance(2,0);
		covarianceMatrix[1][0]=m_ProjectionCovariance(0,1);
		covarianceMatrix[1][1]=m_ProjectionCovariance(1,1);
		covarianceMatrix[1][2]=m_ProjectionCovariance(2,1);
		covarianceMatrix[2][0]=m_ProjectionCovariance(0,2);
		covarianceMatrix[2][1]=m_ProjectionCovariance(1,2);
		covarianceMatrix[2][2]=m_ProjectionCovariance(2,2);
#endif
		covarianceMatrix.Fill(0.0);
		covarianceMatrix[0][0]=m_ProjectionCovariance(0,0);
		covarianceMatrix[1][1]=m_ProjectionCovariance(1,1);
		covarianceMatrix[2][2]=m_ProjectionCovariance(2,2);

		double variance = (m_ProjectionCovariance(0,0) + m_ProjectionCovariance(1,1) + m_ProjectionCovariance(2,2))/3;

		vnl_symmetric_eigensystem<typename MatrixType::ComponentType> eigensystem(covarianceMatrix.GetVnlMatrix());

		//		std::cout << "Covariance" << std::endl;
		//		std::cout << covarianceMatrix << std::endl;

		//		std::cout << "Determinant" << eigensystem.determinant() <<  std::endl;

		//		std::cout << "PInverse" << eigensystem.pinverse() <<  std::endl;


//		std::cout << eigensystem.pinverse()*covarianceMatrix.GetVnlMatrix()<< std::endl;


//		vnl_matrix_inverse<double> invConv(covarianceMatrix.GetVnlMatrix());
//		std::cout << invConv.inverse()*covarianceMatrix.GetVnlMatrix() << std::endl;

//		vnl_cholesky chol(covarianceMatrix.GetVnlMatrix());

//		std::cout << chol.inverse()*covarianceMatrix.GetVnlMatrix() << std::endl;

#if 1


		//		MatrixType invCovariance(eigensystem.pinverse());


		vnl_matrix_inverse<double> invConv(covarianceMatrix.GetVnlMatrix());
		std::cout << invConv.inverse()*covarianceMatrix.GetVnlMatrix() << std::endl;

		MatrixType invCovariance(invConv.inverse());





		class PointAndWeight{
		public:
			PointType point;
			double weight;
			bool operator<(const PointAndWeight & other)const{
				return weight > other.weight;
			}
		};

		std::vector<PointAndWeight> m_Points;

		while(!iterator.IsAtEnd()){

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),point);

			PointType projection = m_Center + (point - m_Center)*m_Radius;

			VectorType diff = (point-projection);



			auto dist =std::exp(-0.5*diff*(invCovariance*diff));
			//erauto dist =std::exp(-(diff*diff)/(2*variance));
			//dist = dist/sqrt(pow(2*M_PI,1.5)*variance);
			//std::cout <<  dist << std::endl;
			PointAndWeight pw;
			pw.point = projection;
			pw.weight=iterator.Get()*dist;

			m_Points.push_back(pw);
			++iterator;

		}

		std::sort(m_Points.begin(),m_Points.end());

		typedef itk::Statistics::Histogram<double> HistogramType;

		  HistogramType::Pointer histogram = HistogramType::New();

		HistogramType::SizeType	 size(1);

		unsigned int bins=1000;
		size.Fill(bins);


		  HistogramType::MeasurementVectorType lowerBound;
		  lowerBound.SetSize(bins);
		  lowerBound.Fill(m_Points[m_Points.size()-1].weight);

		  HistogramType::MeasurementVectorType upperBound;
		  upperBound.SetSize(bins);
		  upperBound.Fill(m_Points[0].weight);

		  histogram->SetMeasurementVectorSize(1);
		  histogram->Initialize(size, lowerBound, upperBound );

		  unsigned long bin=bins-1;
		  for(int k=0;k<m_Points.size();k++){
			  double binMin = histogram->GetBinMin(0,bin);
			  double weight = m_Points[k].weight;
			  while(m_Points[k].weight<histogram->GetBinMin(0,bin)){
				  bin--;
			  }
			  histogram->IncreaseFrequency(bin,1);
		  }
		  //histogram->Print(std::cout,itk::Indent());

		typedef itk::OtsuThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		  //typedef itk::RenyiEntropyThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		//typedef itk::LiThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		//typedef itk::IsoDataThresholdCalculator<HistogramType,double> ThresholdCalculatorType;


		ThresholdCalculatorType::Pointer calculator= ThresholdCalculatorType::New();

		calculator->SetInput(histogram);
		calculator->Update();
		double threshold = calculator->GetThreshold();
		std::cout << "Threshold: " << threshold << std::endl;


		typedef itk::Mesh<double,3> MeshType;

		int k=0;
		while(m_Points[k].weight>threshold){
			this->GetOutput()->SetPoint(k,m_Points[k].point);
			this->GetOutput()->SetPointData(k,m_Points[k].weight);
			k++;
		}



#if 0
		vnl_vector<double> weights(m_Points.size());

		for(int k=0;k<m_Points.size();k++){
			weights[k]=m_Points[k].weight;
		}

		vnl_matlab_filewrite filewrite("./weights.mat");
		filewrite.write(weights,"W");
#endif

		//3. Compute covariance
#endif
	}


protected:
	SphereFittingCalculator(){
		m_Count=0;

	}
	virtual ~SphereFittingCalculator(){

	}

private:
	SphereFittingCalculator(const Self & other);
	void operator=(const Self & other);


	PointType m_CenterOfMass;
	PointType m_Center=m_CenterOfMass;
	double m_Radius;

	double m_TotalIntensity;
	double m_Count;

	SymmetricMatrixType m_ProjectionCovariance;


};
}
int main(int argc, char ** argv) {

	typedef itk::Image<double, 3> ImageType;
	typedef itk::Mesh<double,3> MeshType;

	typedef itk::ImageFileReader<ImageType> ImageFileReaderType;

	ImageFileReaderType::Pointer imageFileReader = ImageFileReaderType::New();

	imageFileReader->SetFileName(argv[1]);
	imageFileReader->Update();

	ImageType::Pointer image = imageFileReader->GetOutput();

	typedef ttt::SphereFittingCalculator<ImageType,MeshType> SphereFittingCalculatorType;

	SphereFittingCalculatorType::Pointer planeFitting = SphereFittingCalculatorType::New();

	planeFitting->SetInput(image);

	typedef itk::MeshFileWriter<MeshType> MeshWriterType;
	MeshWriterType::Pointer meshWriter =MeshWriterType::New();
	meshWriter->SetFileName(argv[2]);
	meshWriter->SetInput(planeFitting->GetOutput());
	meshWriter->Update();

}
