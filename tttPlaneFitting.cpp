#include <itkImageToMeshFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
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
#include <itkWatershedImageFilter.h>
#include <itkMeshToMeshFilter.h>
#include <itkSymmetricSecondRankTensor.h>
#include <itkHessianToObjectnessMeasureImageFilter.h>
#include <itkMultiScaleHessianBasedMeasureImageFilter.h>

#include <itkPointsLocator.h>

#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <vnl/algo/vnl_cholesky.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/vnl_matlab_filewrite.h>

#include <fstream>
namespace ttt {
template<class TInputImage, class TOutputMesh> class PlaneFittingCalculator: public itk::ImageToMeshFilter<
		TInputImage, TOutputMesh> {
public:
	typedef PlaneFittingCalculator Self;
	typedef itk::Object Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self> ConstPointer;

	itkNewMacro(Self)
	;itkTypeMacro(PlaneFittingCalculator,itk::Object)
	;

	typedef TInputImage InputImageType;
	typedef typename TInputImage::ConstPointer ImageConstPointer;
	typedef typename InputImageType::PointType PointType;

	typedef typename PointType::VectorType VectorType;

	typedef typename itk::SymmetricSecondRankTensor<
			typename PointType::CoordRepType, PointType::Dimension> SymmetricMatrixType;
	typedef typename itk::Matrix<typename PointType::CoordRepType,
			PointType::Dimension, PointType::Dimension> MatrixType;

	typedef TOutputMesh OutputMeshType;

	SymmetricMatrixType outerProduct(const VectorType & a,
			const VectorType & b) {
		SymmetricMatrixType matrix;
		for (int r = 0; r < PointType::Dimension; r++) {
			for (int c = 0; c < PointType::Dimension; c++) {
				matrix(r, c) = a[r] * b[c];
			}
		}
		return matrix;
	}

	void ComputeCenterOfMass() {
		//1. Compute center of mass
		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(),
				this->GetInput()->GetLargestPossibleRegion());

		PointType origin = this->GetInput()->GetOrigin();
		VectorType sum;
		sum.Fill(0);
		m_TotalIntensity = 0;
		m_Count = 0;
		while (!iterator.IsAtEnd()) {

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),
					point);

			VectorType vector = (point - origin);
			double weight = iterator.Get();
			sum += weight * vector;
			//std::cout << weight << "\t" << vector << "\t" << sum << std::endl;
			m_TotalIntensity += weight;
			++iterator;
			m_Count++;
		}
		sum = sum / (m_TotalIntensity);

		m_CenterOfMass = origin + sum;
		std::cout << m_CenterOfMass << std::endl;
	}

	void ComputeWeightedImageCovariance() {

		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(),
				this->GetInput()->GetLargestPossibleRegion());

		m_WeightedVolumeCovariance.Fill(0);
		//2. Compute linear fitting

		iterator.GoToBegin();

		while (!iterator.IsAtEnd()) {

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),
					point);

			VectorType vector = (point - m_CenterOfMass);

			m_WeightedVolumeCovariance += this->outerProduct(vector, vector)
					* iterator.Get();

			++iterator;
		}
		m_WeightedVolumeCovariance /= (m_TotalIntensity);
		std::cout << m_WeightedVolumeCovariance << std::endl;
	}

	void ComputeImagePrincipalDirections() {

		m_WeightedVolumeCovariance.ComputeEigenAnalysis(m_EigenValues,
				m_PrincipalDirections);

		std::cout << "eigenValues" << std::endl;
		std::cout << m_EigenValues << std::endl;

		std::cout << "eigenVectors" << std::endl;
		std::cout << m_PrincipalDirections << std::endl;

		m_LeastSquaresPlane[0] = m_PrincipalDirections[0][0];
		m_LeastSquaresPlane[1] = m_PrincipalDirections[0][1];
		m_LeastSquaresPlane[2] = m_PrincipalDirections[0][2];
	}
	void ComputeProjectionCovariance() {

		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(),
				this->GetInput()->GetLargestPossibleRegion());

		m_ProjectionCovariance.Fill(0);
		while (!iterator.IsAtEnd()) {

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),
					point);

			PointType projection = point
					- ((point - m_CenterOfMass) * m_LeastSquaresPlane)
							* m_LeastSquaresPlane;

			//std::cout << point << "\t" << projection << std::endl;

			m_ProjectionCovariance += this->outerProduct(point - projection,
					point - projection) * iterator.Get();

			++iterator;
		}
		m_ProjectionCovariance /= (m_TotalIntensity);
	}
	virtual void GenerateOutputInformation() ITK_OVERRIDE{

	}
	void GenerateData() {

		this->ComputeCenterOfMass();
		this->ComputeWeightedImageCovariance();
		this->ComputeImagePrincipalDirections();
		this->ComputeProjectionCovariance();

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
		std::cout << "Covariance" << std::endl;
		std::cout << covarianceMatrix << std::endl;
#if 1
		covarianceMatrix.Fill(0.0);
		covarianceMatrix[0][0] = m_ProjectionCovariance(0, 0);
		covarianceMatrix[1][1] = m_ProjectionCovariance(1, 1);
		covarianceMatrix[2][2] = m_ProjectionCovariance(2, 2);
#endif
		//double variance = (m_ProjectionCovariance(0,0) + m_ProjectionCovariance(1,1) + m_ProjectionCovariance(2,2))/3;

		//vnl_symmetric_eigensystem<typename MatrixType::ComponentType> eigensystem(covarianceMatrix.GetVnlMatrix());

		//		std::cout << "Covariance" << std::endl;
		//		std::cout << covarianceMatrix << std::endl;

		//		std::cout << "Determinant" << eigensystem.determinant() <<  std::endl;

		//		std::cout << "PInverse" << eigensystem.pinverse() <<  std::endl;

		//std::cout << eigensystem.pinverse()*covarianceMatrix.GetVnlMatrix()<< std::endl;

		vnl_matrix_inverse<double> invConv(covarianceMatrix.GetVnlMatrix());
		std::cout << invConv.inverse() * covarianceMatrix.GetVnlMatrix()
				<< std::endl;

		//vnl_cholesky chol(covarianceMatrix.GetVnlMatrix());

		//std::cout << chol.inverse()*covarianceMatrix.GetVnlMatrix() << std::endl;

#if 1

		MatrixType invCovariance(invConv.inverse());

		class PointAndWeight {
		public:
			PointType point;
			double weight;
			bool operator<(const PointAndWeight & other) const {
				return weight > other.weight;
			}
		};

		std::vector<PointAndWeight> m_Points;
		typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> IteratorType;

		IteratorType iterator(this->GetInput(),
				this->GetInput()->GetLargestPossibleRegion());

		while (!iterator.IsAtEnd()) {

			PointType point;
			this->GetInput()->TransformIndexToPhysicalPoint(iterator.GetIndex(),
					point);

			PointType projection = point
					- ((point - m_CenterOfMass) * m_LeastSquaresPlane)
							* m_LeastSquaresPlane;

			VectorType diff = (point - projection);

			auto dist = std::exp(-0.5 * diff * (invCovariance * diff));
			//auto dist =std::exp(-(diff*diff)/(2*variance));
			//dist = dist/sqrt(pow(2*M_PI,1.5)*variance);
			//std::cout <<  dist << std::endl;
			PointAndWeight pw;
			pw.point = projection;
			pw.weight = iterator.Get() * dist;

			m_Points.push_back(pw);
			++iterator;

		}

		std::sort(m_Points.begin(), m_Points.end());

		typedef itk::Statistics::Histogram<double> HistogramType;

		HistogramType::Pointer histogram = HistogramType::New();

		HistogramType::SizeType size(1);

		unsigned int bins = 1000;
		size.Fill(bins);

		HistogramType::MeasurementVectorType lowerBound;
		lowerBound.SetSize(bins);
		lowerBound.Fill(m_Points[m_Points.size() - 1].weight);

		HistogramType::MeasurementVectorType upperBound;
		upperBound.SetSize(bins);
		upperBound.Fill(m_Points[0].weight);

		histogram->SetMeasurementVectorSize(1);
		histogram->Initialize(size, lowerBound, upperBound);

		unsigned long bin = bins - 1;
		for (int k = 0; k < m_Points.size(); k++) {
			double binMin = histogram->GetBinMin(0, bin);
			double weight = m_Points[k].weight;
			while (m_Points[k].weight < histogram->GetBinMin(0, bin)) {
				bin--;
			}
			histogram->IncreaseFrequency(bin, 1);
		}
		//histogram->Print(std::cout,itk::Indent());

		typedef itk::OtsuThresholdCalculator<HistogramType, double> ThresholdCalculatorType;
		//typedef itk::RenyiEntropyThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		//typedef itk::LiThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		//typedef itk::IsoDataThresholdCalculator<HistogramType,double> ThresholdCalculatorType;

		ThresholdCalculatorType::Pointer calculator =
				ThresholdCalculatorType::New();

		calculator->SetInput(histogram);
		calculator->Update();
		double threshold = calculator->GetThreshold();
		std::cout << "Threshold: " << threshold << std::endl;
		iterator.GoToBegin();
		int k = 0;

		while (!iterator.IsAtEnd()) {
			if (iterator.Get() > threshold) {
				PointType projection = this->m_PrincipalDirections
						* m_Points[k].point;
				this->GetOutput()->SetPoint(k, projection);
				this->GetOutput()->SetPointData(k, m_Points[k].weight);
				k++;
			}

			++iterator;
		}
#if 0
		int k = 0;
		while (m_Points[k].weight > threshold) {
			this->GetOutput()->SetPoint(k, m_Points[k].point);
			this->GetOutput()->SetPointData(k, m_Points[k].weight);
			k++;
		}
#endif
#if 0
		vnl_vector<double> weights(m_Points.size());

		for(int k=0;k<m_Points.size();k++) {
			weights[k]=m_Points[k].weight;
		}

		vnl_matlab_filewrite filewrite("./weights.mat");
		filewrite.write(weights,"W");
#endif

		//3. Compute covariance
#endif
	}

protected:
	PlaneFittingCalculator() {
		m_Count = 0;

	}
	virtual ~PlaneFittingCalculator() {

	}

private:
	PlaneFittingCalculator(const Self & other);
	void operator=(const Self & other);

	PointType m_CenterOfMass;
	SymmetricMatrixType m_WeightedVolumeCovariance;
	double m_TotalIntensity;
	double m_Count;

	VectorType m_LeastSquaresPlane;
	MatrixType m_PrincipalDirections;
	VectorType m_EigenValues;
	SymmetricMatrixType m_ProjectionCovariance;

};


template<class TInputMesh,class TOutputMesh> class PointCloudThresholdFilter : public itk::MeshToMeshFilter<TInputMesh,TOutputMesh>{
public:
	typedef PointCloudThresholdFilter Self;
	typedef itk::MeshToMeshFilter<TInputMesh,TOutputMesh> Superclass;

	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self> ConstPointer;

	typedef typename TInputMesh::PointType PointType;
	itkNewMacro(Self);

protected:
	PointCloudThresholdFilter(){

	}
	virtual ~PointCloudThresholdFilter(){

	}
	virtual void GenerateData() ITK_OVERRIDE{

		class PointAndWeight {
		public:
			PointType point;
			double weight;
			bool operator<(const PointAndWeight & other) const {
				return weight > other.weight;
			}
		};

		std::vector<PointAndWeight> m_Points;


		for(auto pointIt=this->GetInput()->GetPoints()->Begin();pointIt!=this->GetInput()->GetPoints()->End();++pointIt){
			PointAndWeight pw;

			this->GetInput()->GetPointData(pointIt.Index(),&pw.weight);
			pw.point=pointIt.Value();

			m_Points.push_back(pw);
		}

		std::sort(m_Points.begin(), m_Points.end());

		typedef itk::Statistics::Histogram<double> HistogramType;

		HistogramType::Pointer histogram = HistogramType::New();

		HistogramType::SizeType size(1);

		unsigned int bins = 1000;
		size.Fill(bins);

		HistogramType::MeasurementVectorType lowerBound;
		lowerBound.SetSize(bins);
		lowerBound.Fill(m_Points[m_Points.size() - 1].weight);

		HistogramType::MeasurementVectorType upperBound;
		upperBound.SetSize(bins);
		upperBound.Fill(m_Points[0].weight);

		histogram->SetMeasurementVectorSize(1);
		histogram->Initialize(size, lowerBound, upperBound);

		unsigned long bin = bins - 1;
		for (int k = 0; k < m_Points.size(); k++) {
			double binMin = histogram->GetBinMin(0, bin);
			double weight = m_Points[k].weight;
			while (m_Points[k].weight < histogram->GetBinMin(0, bin)) {
				bin--;
			}
			histogram->IncreaseFrequency(bin, 1);
		}
		//histogram->Print(std::cout,itk::Indent());

		typedef itk::OtsuThresholdCalculator<HistogramType, double> ThresholdCalculatorType;
		//typedef itk::RenyiEntropyThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		//typedef itk::LiThresholdCalculator<HistogramType,double> ThresholdCalculatorType;
		//typedef itk::IsoDataThresholdCalculator<HistogramType,double> ThresholdCalculatorType;

		ThresholdCalculatorType::Pointer calculator =
				ThresholdCalculatorType::New();

		calculator->SetInput(histogram);
		calculator->Update();
		double threshold = calculator->GetThreshold();
		std::cout << "Threshold: " << threshold << std::endl;

		for(int k=0;k<m_Points.size();k++){
			if(m_Points[k].weight>threshold){
				this->GetOutput()->SetPoint(k,m_Points[k].point);
				this->GetOutput()->SetPointData(k,m_Points[k].weight);
			}
		}

	}
private:
	PointCloudThresholdFilter(const Self & other);
	void operator=(const Self & other);
};


template<class TInputMesh,class TOutputMesh> class PointCloudObjectnessFeatureImageFilter : public itk::MeshToMeshFilter<TInputMesh,TOutputMesh>{

public:
	typedef PointCloudObjectnessFeatureImageFilter Self;
	typedef itk::MeshToMeshFilter<TInputMesh,TOutputMesh> Superclass;

	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self> ConstPointer;

	typedef typename TInputMesh::PointType PointType;
	itkNewMacro(Self);

	typedef	itk::SymmetricSecondRankTensor<double,3> SymmetricMatrixType;
	typedef itk::Vector<double,3> VectorType;

	itkGetMacro(Radius,double)
	itkSetMacro(Radius,double)
protected:

	PointCloudObjectnessFeatureImageFilter(){

		m_Radius=1;

	}
	SymmetricMatrixType outerProduct(const VectorType & a,
			const VectorType & b) {
		SymmetricMatrixType matrix;
		for (int r = 0; r < PointType::Dimension; r++) {
			for (int c = 0; c < PointType::Dimension; c++) {
				matrix(r, c) = a[r] * b[c];
			}
		}
		return matrix;
	}
	template<class TIterator> SymmetricMatrixType ComputeLocalTensor(const PointType & reference,const TIterator & begin,const TIterator & end){

		SymmetricMatrixType matrix;
		int totalPoints =0;
		for(auto it=begin;it!=end;++it){
			auto point = this->GetInput()->GetPoint(*it);
			double weight;
			this->GetInput()->GetPointData(*it,&weight);
			auto vector = reference - point;
			matrix=matrix+outerProduct(vector,vector)*weight;
			totalPoints++;
		}
		return matrix/totalPoints;

	}
	virtual void GenerateData() ITK_OVERRIDE{
		typename TInputMesh::Pointer input =const_cast<TInputMesh*>(this->GetInput());
		m_PointsLocator = PointLocatorType::New();
		m_PointsLocator->SetPoints(input->GetPoints());

		m_PointsLocator->Initialize();
		int k=0;
		for(auto  pointIt =input->GetPoints()->Begin();pointIt!=input->GetPoints()->End();++pointIt){

			NeighborsIdentifierType neighbors;
			m_PointsLocator->FindPointsWithinRadius(pointIt->Value(),m_Radius,neighbors);
			SymmetricMatrixType localTensor=this->ComputeLocalTensor(pointIt->Value(),neighbors.begin(),neighbors.end());
			SymmetricMatrixType::EigenValuesArrayType eigenvalues;
			localTensor.ComputeEigenValues(eigenvalues);


			if(eigenvalues[0]<0){

			}else if(eigenvalues[0]==0 && eigenvalues[1]>=0){
				double weight;
				this->GetInput()->GetPointData(pointIt->Index(),&weight);
				double value = weight*(1- std::sqrt(1- std::pow(eigenvalues[1],2)/std::pow(eigenvalues[2],2)));
				this->GetOutput()->SetPointData(k,value);
				this->GetOutput()->SetPoint(k,pointIt->Value());
				k++;
			}
		}

#if 0
		std::for_each(m_PointLocator->GetPoints()->Begin(),m_PointLocator->GetPoints()->End(),
				[&](TInputMesh::PointsContainer::){

		});
#endif
	}
private:


	typedef itk::PointsLocator<typename TInputMesh::PointsContainer> PointLocatorType;
	typename PointLocatorType::Pointer m_PointsLocator;
	typedef typename PointLocatorType::NeighborsIdentifierType NeighborsIdentifierType;


	double m_Radius;


};

template<class TInputMesh, class TOutputImage> class RasterMeshImageFilter: public itk::ImageSource<
		TOutputImage> {

public:
	typedef RasterMeshImageFilter Self;
	typedef itk::ImageSource<TOutputImage> Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self> ConstPointer;

	itkNewMacro(Self)
	;itkTypeMacro(RasterMeshImageFilter,TOutputImage)
	;
	typedef TInputMesh InputMeshType;
	typedef typename TInputMesh::Pointer InputMeshPointer;

	typedef TOutputImage OutputImageType;
	typedef typename TOutputImage::Pointer OutputImagePointer;

	using Superclass::SetInput;

	void SetInput(const TInputMesh * mesh) {
		this->itk::ProcessObject::SetNthInput(0, const_cast<TInputMesh*>(mesh));
	}
	const TInputMesh * GetInput() {
		return dynamic_cast<const TInputMesh*>(this->itk::ProcessObject::GetInput(
				0));

	}
protected:


	RasterMeshImageFilter() {
		this->SetNumberOfRequiredInputs(1);

//		OutputImagePointer output = dynamic_cast<OutputImageType*>(this->MakeOutput(0).GetPointer())
	}
	virtual void GenerateOutputInformation() ITK_OVERRIDE {
#if 0
		typedef typename TOutputImage::RegionType RegionType;
		typedef typename TOutputImage::SizeType SizeType;
		typedef typename TOutputImage::IndexType IndexType;

		RegionType region;
		IndexType index;
		index.Fill(0);
		SizeType size;
		size.Fill(0);
		region.SetIndex(index);
		region.SetSize(size);
		this->GetOutput()->SetRegions(region);
#endif

		InputMeshPointer input = const_cast<InputMeshType*>(this->GetInput());
		input->Update();
		m_MinX = itk::NumericTraits<double>::max();
		m_MinY = itk::NumericTraits<double>::max();
		m_MaxX = itk::NumericTraits<double>::min();
		m_MaxY = itk::NumericTraits<double>::min();

		for (auto pointIterator = input->GetPoints()->Begin();
				pointIterator != input->GetPoints()->End(); pointIterator++) {
			if (pointIterator->Value()[1] < m_MinX)
				m_MinX = pointIterator->Value()[1];
			if (pointIterator->Value()[2] < m_MinY)
				m_MinY = pointIterator->Value()[2];

			if (pointIterator->Value()[1] > m_MaxX)
				m_MaxX = pointIterator->Value()[1];
			if (pointIterator->Value()[2] > m_MaxY)
				m_MaxY = pointIterator->Value()[2];
		}
		std::cout << "Bounds:" << m_MinX << "\t" << m_MinY << "\t" << m_MaxX
				<< "\t" << m_MaxY << std::endl;

		m_RangeX = m_MaxX - m_MinX;
		m_RangeY = m_MaxY - m_MinY;

		int sizeX = 256;
		int sizeY = 256;


		double spacingX = m_RangeX / sizeX;
		double spacingY = m_RangeY / sizeY;

		typename OutputImageType::RegionType region;
		typename OutputImageType::SizeType size;
		typename OutputImageType::PointType origin;

		typename OutputImageType::IndexType index;
		typename OutputImageType::SpacingType spacing;

		spacing[0] = spacingX;
		spacing[1] = spacingY;

		origin[0] = m_MinX;
		origin[1] = m_MinY;

		size[0] = sizeX;
		size[1] = sizeY;
		index.Fill(0.0);

		this->GetOutput()->SetOrigin(origin);
		this->GetOutput()->SetSpacing(spacing);
		region.SetIndex(index);
		region.SetSize(size);
		this->GetOutput()->SetRegions(region);
	}
	virtual void EnlargeOutputRequestedRegion(itk::DataObject * output)
			ITK_OVERRIDE {
		//output->SetRequestedRegionToLargestPossibleRegion();
	}

	void GenerateData() {
		std::cout << "Rastering" << std::endl;
		this->GetOutput()->Allocate();
		this->GetOutput()->FillBuffer(0);

		for (auto pointIterator = this->GetInput()->GetPoints()->Begin();
				pointIterator != this->GetInput()->GetPoints()->End();
				pointIterator++) {
			typename OutputImageType::IndexType index;
			typename OutputImageType::PointType point;
			point[0] = pointIterator->Value()[1];
			point[1] = pointIterator->Value()[2];
			this->GetOutput()->TransformPhysicalPointToIndex(point, index);
			double value;

			this->GetInput()->GetPointData(pointIterator->Index(), &value);
			//std::cout << index << "\t" << this->GetOutput()->GetPixel(index) << "\t" << this->GetOutput()->GetPixel(index)+value << std::endl;
			this->GetOutput()->SetPixel(index,
					this->GetOutput()->GetPixel(index) + value);
		}
	}

private:
	RasterMeshImageFilter(const Self & other);
	void operator=(const Self & other);

	double m_MinX;
	double m_MaxX;

	double m_MinY;
	double m_MaxY;

	double m_RangeX;
	double m_RangeY;
};
}		//namespace ttt
int main(int argc, char ** argv) {

	typedef itk::Image<double, 3> ImageType;
	typedef itk::Image<float, 2> ImageType2D;
	typedef itk::Mesh<double, 3> MeshType;

	typedef itk::ImageFileReader<ImageType> ImageFileReaderType;

	ImageFileReaderType::Pointer imageFileReader = ImageFileReaderType::New();

	imageFileReader->SetFileName(argv[1]);
	imageFileReader->Update();

	ImageType::Pointer image = imageFileReader->GetOutput();

	typedef ttt::PlaneFittingCalculator<ImageType, MeshType> PlaneFittingCalculatorType;

	PlaneFittingCalculatorType::Pointer planeFitting =
			PlaneFittingCalculatorType::New();

	planeFitting->SetInput(image);


	typedef itk::MeshFileWriter<MeshType> MeshWriterType;
	MeshWriterType::Pointer meshWriter = MeshWriterType::New();
	meshWriter->SetFileName(argv[2]);
	meshWriter->SetInput(planeFitting->GetOutput());
	meshWriter->Update();



	typedef ttt::PointCloudObjectnessFeatureImageFilter<MeshType,MeshType> PointCloudObjectnessFeatureType;

	typename PointCloudObjectnessFeatureType::Pointer meshFeatureFilter=PointCloudObjectnessFeatureType::New();
	double radius =atof(argv[4]);
	meshFeatureFilter->SetRadius(radius);
	meshFeatureFilter->SetInput(planeFitting->GetOutput());

	typedef ttt::PointCloudThresholdFilter<MeshType,MeshType> ThresholdFilterType;

	ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();
	thresholdFilter->SetInput(meshFeatureFilter->GetOutput());

	typedef itk::MeshFileWriter<MeshType> MeshWriterType;
	MeshWriterType::Pointer meshFeatureWriter = MeshWriterType::New();
	meshFeatureWriter->SetFileName(argv[3]);
	meshFeatureWriter->SetInput(thresholdFilter->GetOutput());
	meshFeatureWriter->Update();



#if 0
	typedef itk::MeshFileWriter<MeshType> MeshWriterType;
	MeshWriterType::Pointer meshWriter = MeshWriterType::New();
	meshWriter->SetFileName(argv[2]);
	meshWriter->SetInput(planeFitting->GetOutput());
	meshWriter->Update();
#endif
#if 0
	typedef ttt::RasterMeshImageFilter<MeshType, ImageType2D> RasterType;
	RasterType::Pointer raster = RasterType::New();
	raster->SetInput(planeFitting->GetOutput());


	  typedef itk::SymmetricSecondRankTensor< double, 2 > HessianPixelType;
	  typedef itk::Image< HessianPixelType, 2 >           HessianImageType;
	  typedef itk::HessianToObjectnessMeasureImageFilter< HessianImageType, ImageType2D >
	    ObjectnessFilterType;
	  ObjectnessFilterType::Pointer objectnessFilter = ObjectnessFilterType::New();
	  objectnessFilter->SetBrightObject( true );
	  objectnessFilter->SetScaleObjectnessMeasure( false );
	  objectnessFilter->SetAlpha( 0.5 );
	  objectnessFilter->SetBeta( 1.0 );
	  objectnessFilter->SetGamma( 5.0 );

	  int numberOfSigmaSteps=4;
	  double sigmaMinimum=0.5;
	  double sigmaMaximum=1.0;

	  typedef itk::MultiScaleHessianBasedMeasureImageFilter< ImageType2D, HessianImageType, ImageType2D >
	    MultiScaleEnhancementFilterType;
	  MultiScaleEnhancementFilterType::Pointer multiScaleEnhancementFilter =
	    MultiScaleEnhancementFilterType::New();
	  multiScaleEnhancementFilter->SetInput( raster->GetOutput() );
	  multiScaleEnhancementFilter->SetHessianToMeasureFilter( objectnessFilter );
	  multiScaleEnhancementFilter->SetSigmaStepMethodToLogarithmic();
	  multiScaleEnhancementFilter->SetSigmaMinimum( sigmaMinimum );
	  multiScaleEnhancementFilter->SetSigmaMaximum( sigmaMaximum );
	  multiScaleEnhancementFilter->SetNumberOfSigmaSteps( numberOfSigmaSteps );





	typedef itk::WatershedImageFilter<ImageType2D> WatershedFilterType;

	WatershedFilterType::Pointer watershed = WatershedFilterType::New();
	watershed->SetLevel(0.4);
	watershed->SetThreshold(0.005);

	watershed->SetInput(multiScaleEnhancementFilter->GetOutput());

	typedef itk::ImageFileWriter<RasterType::OutputImageType> ImageWriterType;

	ImageWriterType::Pointer imageWriter = ImageWriterType::New();
	imageWriter->SetInput(multiScaleEnhancementFilter->GetOutput());
	imageWriter->SetFileName(argv[3]);
	imageWriter->Update();
#endif
#if 0
	typedef itk::ImageFileWriter<WatershedFilterType::OutputImageType> ImageWriterType;

	ImageWriterType::Pointer imageWriter = ImageWriterType::New();
	imageWriter->SetInput(watershed->GetOutput());
	imageWriter->SetFileName(argv[3]);
	imageWriter->Update();
#endif
}

