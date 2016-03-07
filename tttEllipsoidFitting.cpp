
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
#include <vnl/algo/vnl_solve_qp.h>
#include <vnl/vnl_matlab_filewrite.h>

//#include <cminpack.h>
#include <fstream>
namespace ttt {
template<class TInputImage,class TOutputMesh> class EllipsoidFittingCalculator: public itk::ImageToMeshFilter<TInputImage,TOutputMesh> {
public:
	typedef EllipsoidFittingCalculator Self;
	typedef itk::Object Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self> ConstPointer;

	itkNewMacro(Self)
	;itkTypeMacro(EllipsoidFittingCalculator,itk::Object)
	;

	typedef TInputImage InputImageType;
	typedef typename TInputImage::ConstPointer ImageConstPointer;
	typedef typename InputImageType::PointType PointType;

	typedef typename PointType::VectorType VectorType;

	typedef typename itk::SymmetricSecondRankTensor<typename PointType::CoordRepType,PointType::Dimension> SymmetricMatrixType;
	typedef typename itk::Matrix<typename PointType::CoordRepType,PointType::Dimension,PointType::Dimension> MatrixType;


	template<class TVector,class TMatrix> TMatrix outerProduct(const TVector & a, const TVector & b){
		TMatrix matrix;
		for(int r=0;r<TVector::Dimension;r++){
			for(int c=0;c<TVector::Dimension;c++){
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

	void ComputeEllipsoidFitting(){

		typedef vnl_matrix_fixed<double,10,10> ScatterMatrixType;
		typedef vnl_vector_fixed<double,10> AugmentedVectorType;

		typedef vnl_matrix_fixed<double,6,6> C1Type;

		C1Type C1;
		C1.fill(0.0);

		C1[0][0]=-1;
		C1[0][1]=1;
		C1[0][2]=1;

		C1[1][0]=1;
		C1[1][1]=-1;
		C1[1][2]=1;

		C1[2][0]=1;
		C1[2][1]=1;
		C1[2][2]=-1;

		C1[3][3]=-4;
		C1[4][4]=-4;
		C1[5][5]=-4;


		ScatterMatrixType DDt;
		DDt.fill(0.0);

		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;


		IteratorType iterator(this->GetInput(), this->GetInput()->GetLargestPossibleRegion());

		while (!iterator.IsAtEnd()) {

			PointType point;

			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(), point);
			point = point - m_CenterOfMass;

			AugmentedVectorType x;
			x[0]=std::pow(point[0],2);
			x[1]=std::pow(point[1],2);
			x[2]=std::pow(point[2],2);
			x[3]=2*point[1]*point[2];
			x[4]=2*point[0]*point[2];
			x[5]=2*point[0]*point[1];
			x[6]=2*point[0];
			x[7]=2*point[1];
			x[8]=2*point[2];
			x[9]=1;

			double weight = iterator.Get();
			if(weight<0) weight=0;
			DDt+=outer_product(x,x);
			//DDt+= this->outerProduct<AugmentedVectorType,ScatterMatrixType>(x,x)*weight;

			++iterator;
		}

		typedef vnl_matrix_fixed<double,6,6> S11Type;
		typedef vnl_matrix_fixed<double,6,4> S12Type;
		typedef vnl_matrix_fixed<double,4,4> S22Type;


		S11Type s11=DDt.get_n_columns(0,6).get_n_rows(0,6);
		S12Type s12=DDt.get_n_columns(6,4).get_n_rows(0,6);
		S22Type s22=DDt.get_n_columns(6,4).get_n_rows(6,4);

		std::cout << s11 << std::endl;
		std::cout << s12 << std::endl;
		std::cout << s22 << std::endl;

		C1Type invC1 = vnl_matrix_inverse<double>(C1).inverse();

		std::cout << invC1 << std::endl;
		S22Type invS22 = vnl_matrix_inverse<double>(s22).inverse();
		std::cout << invS22 << std::endl;
		vnl_symmetric_eigensystem<double> eigensystem(invC1*(s11-s12*invS22*s12.transpose()));

		std::cout << "____________________"<<std::endl;
		std::cout << eigensystem.D << std::endl;
		std::cout << "____________________"<<std::endl;
		std::cout << eigensystem.V << std::endl;
		std::cout << "____________________"<<std::endl;

		vnl_vector<double> v1 = eigensystem.get_eigenvector(5);

		vnl_vector<double> v2 = -invS22*s12.transpose()*v1;


		double A,B,C,F,G,H,P,Q,R,D;

		A=v1[0];
		B=v1[1];
		C=v1[2];
		F=v1[3];
		G=v1[4];
		H=v1[5];

		P=v2[0];
		Q=v2[1];
		R=v2[2];
		D=v2[3];


		m_A[0][0]=A;
		m_A[1][1]=B;
		m_A[2][2]=C;

		m_A[0][1]=H/2;
		m_A[0][2]=G/2;
		m_A[1][0]=H/2;
		m_A[1][2]=F/2;
		m_A[2][0]=G/2;
		m_A[2][1]=F/2;

		m_Center[0]=-(4*B*C*P-2*B*G*R-2*C*H*Q-F*F*P + F*G*Q+ F*H*R)/(2*(4*A*B*C-A*F*F - B*G*G-C*H*H +F*G*H));
		m_Center[1]=-(4*A*C*Q-2*A*F*R-2*C*H*P+F*G*P -G*G*Q + G*H*R)/(2*(4*A*B*C-A*F*F - B*G*G-C*H*H +F*G*H));
		m_Center[2]=-(4*A*B*R-2*A*F*Q-2*B*G*P-F*H*P - G*H*Q -H*H*R)/(2*(4*A*B*C-A*F*F - B*G*G-C*H*H +F*G*H));

		m_Center+=m_CenterOfMass.GetVectorFromOrigin();


		std::cout << m_A << std::endl;
		std::cout << m_Center << std::endl;


		std::cout << v1 << std::endl;
		std::cout << v2 << std::endl;


	}

	void ComputeProjectionCovariance(){


		vnl_cholesky cholA(m_A.GetVnlMatrix());
		vnl_matrix<double> Lt = cholA.upper_triangle();
		vnl_matrix<double> invLt = vnl_matrix_inverse<double>(Lt).inverse();

		//std::cout << invA*m_A<<std::endl;


		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(),this->GetInput()->GetLargestPossibleRegion());

		m_ProjectionCovariance.Fill(0);
		int k=0;
		while(!iterator.IsAtEnd()){

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),point);


			auto g = (point-m_Center) -m_CenterOfMass.GetVectorFromOrigin();
			vnl_vector<double> y =  m_A*g.GetVnlVector();
			double norm = y.two_norm();
			assert(norm<=1);
			vnl_vector<double> projection=m_Center.GetVnlVector()+ invLt*(y);

			PointType projectedPoint;
			projectedPoint[0]=projection[0];
			projectedPoint[1]=projection[1];
			projectedPoint[2]=projection[2];
			std::cout << g << "\t" << projection<< std::endl;

			this->GetOutput()->SetPoint(k,PointType(projectedPoint));
			//this->GetOutput()->SetPointData(k,m_Points[k].weight);
			k++;

			//m_ProjectionCovariance+=this->outerProduct(point-projection,point-projection)*iterator.Get();

			++iterator;
		}
		m_ProjectionCovariance/=(m_TotalIntensity);
	}
	void GenerateData() {

		this->ComputeCenterOfMass();
		this->ComputeEllipsoidFitting();

		this->ComputeProjectionCovariance();
		return;

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
	EllipsoidFittingCalculator(){
		m_Count=0;

	}
	virtual ~EllipsoidFittingCalculator(){

	}

private:
	EllipsoidFittingCalculator(const Self & other);
	void operator=(const Self & other);


	PointType m_CenterOfMass;
	PointType m_Center;
	double m_Radius;

	double m_TotalIntensity;
	double m_Count;


	MatrixType m_A;
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

	typedef ttt::EllipsoidFittingCalculator<ImageType,MeshType> SphereFittingCalculatorType;

	SphereFittingCalculatorType::Pointer planeFitting = SphereFittingCalculatorType::New();

	planeFitting->SetInput(image);

	typedef itk::MeshFileWriter<MeshType> MeshWriterType;
	MeshWriterType::Pointer meshWriter =MeshWriterType::New();
	meshWriter->SetFileName(argv[2]);
	meshWriter->SetInput(planeFitting->GetOutput());
	meshWriter->Update();

}
