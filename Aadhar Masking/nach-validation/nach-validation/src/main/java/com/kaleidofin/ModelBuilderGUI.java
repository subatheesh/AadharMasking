package com.kaleidofin;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.nd4j.linalg.api.ndarray.INDArray;


import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import boofcv.abst.feature.detect.line.DetectLineHoughPolar;
import boofcv.abst.feature.detect.line.DetectLineSegmentsGridRansac;
import boofcv.abst.filter.binary.BinaryContourFinder;
import boofcv.abst.filter.binary.BinaryContourInterface;
import boofcv.alg.distort.RemovePerspectiveDistortion;
import boofcv.alg.feature.detect.edge.CannyEdge;
import boofcv.alg.feature.detect.edge.EdgeContour;
import boofcv.alg.feature.detect.edge.EdgeSegment;
import boofcv.alg.filter.binary.BinaryImageOps;
import boofcv.alg.filter.binary.Contour;
import boofcv.alg.filter.binary.ContourPacked;
import boofcv.alg.filter.binary.GThresholdImageOps;
import boofcv.alg.filter.binary.ThresholdImageOps;
import boofcv.alg.misc.ImageStatistics;
import boofcv.alg.shapes.ShapeFittingOps;
import boofcv.factory.feature.detect.edge.FactoryEdgeDetectors;
import boofcv.factory.feature.detect.line.ConfigHoughPolar;
import boofcv.factory.feature.detect.line.FactoryDetectLineAlgs;
import boofcv.factory.filter.binary.FactoryBinaryContourFinder;
import boofcv.gui.ListDisplayPanel;
import boofcv.gui.binary.VisualizeBinaryData;
import boofcv.gui.feature.ImageLinePanel;
import boofcv.gui.feature.VisualizeShapes;
import boofcv.gui.image.ShowImages;
import boofcv.io.UtilIO;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.ConnectRule;
import boofcv.struct.PointIndex_I32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayS16;
import boofcv.struct.image.GrayS32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageGray;
import boofcv.struct.image.ImageType;
import boofcv.struct.image.Planar;
import georegression.struct.line.LineParametric2D_F32;
import georegression.struct.line.LineSegment2D_F32;
import georegression.struct.point.Point2D_F64;
import georegression.struct.point.Point2D_I32;

public class ModelBuilderGUI {
    private enum NachField {IFSC_CODE, ACCOUNT_NUMBER, REFERENCE_NUMBER, MONTHLY, QUARTERLY, HALF_YEARLY, YEARLY, FIXED_AMOUNT}
	
	static BufferedImage buff = null;
	
	static BufferedImage flat = null;
	// Used to bias it towards more or fewer sides. larger number = fewer sides
	static double cornerPenalty = 10.0;
	// The fewest number of pixels a side can have
	static int minSide = 50;
	
	static double minimumSideFraction = 0.1;
	
	static GrayU8 binny;
	
	private static final float edgeThreshold = 25;

	private static final int maxLines = 10;
	
	private static ListDisplayPanel lp = new ListDisplayPanel();
	
	private static List<Segment> list2; 
	
	public String bankNumber;
	public static boolean isBankNamePresent;
	public String ifsc;
	public String amountInWords;
	public String amountInNumbers;
	public static boolean isMonthlyTicked;
	public static boolean isquarterlyTicked;
	public static boolean ishalfYearlyTicked;
	public static boolean isyearlyTicked;
	public static boolean isAsAndWhenTicked;
	public static boolean isSignPresent;
	public static boolean isNamePresent;
	
	private static final String NACH_REFERENCE_PATTERN = "K[A-Z]-(\\d{4})-(\\d{8})\\b";
	
	static HashMap<Integer, BufferedImage> hm = new HashMap<Integer, BufferedImage>();
	
	
	static ListDisplayPanel gui2 = new ListDisplayPanel();

    public ModelBuilderGUI() {
        new ListDisplayPanel();
    }

    private static String getReferenceCode(String s) {
        Pattern pattern = Pattern.compile(NACH_REFERENCE_PATTERN);
        Matcher matcher = pattern.matcher(s);
        if (matcher.find()) {
            return matcher.group(); // you can get it from desired index as well
        }
        return null;
    }
    
    
    public void travaille_5_perspective_transformer_optimized(BufferedImage image,List<Point2D_F64> list) {
    	
    	 // SOURCE * * * * :
    	int largueur = image.getWidth();
    	int hauteur = image.getHeight();
    	short image1[][] = new short [hauteur][largueur]; // to set
    	double x1p, x2p, x3p, x4p, y1p, y2p, y3p, y4p; // to set
    	
    	x1p = list.get(0).getX();
    	x2p = list.get(1).getX();
    	x3p = list.get(2).getX();
    	x4p = list.get(3).getX();
    	
    	y1p = list.get(0).getY();
    	y2p = list.get(1).getY();
    	y3p = list.get(2).getY();
    	y4p = list.get(3).getY();
    	

    	// TARGET * * * * :
    	int sudoku_dim = 486;
    	short image2[][] = new short [1600][800];
        
    	            
    	       int x, y; // x, y on page 380
    	       int xp, yp; // x prime, y prime on page 380
    	        
    	       double a11, a12, a13, a21, a22, a23, a31, a32, a33; // on page 383
    	   
    	       double numerateur_x = 0; // to improve performance
    	       double numerateur_y = 0; // to improve performance
    	       double denominateur = 0; // to improve performance  
    	        
    	       double numerateur_x_i = 0; // to improve performance
    	       double numerateur_y_i = 0; // to improve performance
    	       double denominateur_i = 0; // to improve performance  
    	               
    	       short la_ligne []; // to improve performance
    	     
    	       // formula 16.28 on page 383
    	       a31 = ((x1p-x2p+x3p-x4p)*(y4p-y3p)-(y1p-y2p+y3p-y4p)*(x4p-x3p))/((x2p-x3p)*(y4p-y3p)-(x4p-x3p)*(y2p-y3p));
    	    
    	       // formula 16.29 on page 383
    	       a32 = ((y1p-y2p+y3p-y4p)*(x2p-x3p)-(x1p-x2p+x3p-x4p)*(y2p-y3p))/((x2p-x3p)*(y4p-y3p)-(x4p-x3p)*(y2p-y3p));
    	    
    	       // formula 16.17 on page 380
    	       a33 = 1;
    	    
    	       // formula 16.30 on page 383
    	       a11 = x2p-x1p+a31*x2p;
    	       a12 = x4p-x1p+a32*x4p;
    	       a13 = x1p;
    	    
    	       // formula 16.31 on page 383
    	       a21 = y2p-y1p+a31*y2p;
    	       a22 = y4p-y1p+a32*y4p;
    	       a23 = y1p;

    	       // Projective mapping via the unit square on page 382 : unit square to image2 dimension
    	       a31 = a31 / sudoku_dim;
    	       a32 = a32 / sudoku_dim;

    	       a11 = a11 / sudoku_dim;
    	       a12 = a12 / sudoku_dim;

    	       a21 = a21 / sudoku_dim;
    	       a22 = a22 / sudoku_dim;
    	       
    	       // for each point (x, y) in image2,
    	       //      we calculate (xp, yp) in image1 using formula 16.18 and 16.19 on page 380
    	       // the formulas are transformed a bit to improve performance by avoiding many multiplications
    	    
    	       numerateur_x_i = a13;
    	       numerateur_y_i = a23;
    	       denominateur_i = a33;
    	          
    	       for (y = 0; y < 800; y++) {
    	        
    	             numerateur_x = numerateur_x_i;
    	             numerateur_y = numerateur_y_i;
    	             denominateur = denominateur_i;
    	               
    	             la_ligne = image2[y]; // to improve performance

    	             for (x = 0; x < 1600; x++) {

    	                    xp = (int) (numerateur_x/denominateur + 0.5); // formula 16.18 on page 380
    	                    yp = (int) (numerateur_y/denominateur + 0.5); // formula 16.19 on page 380

    	                    if (yp >= 0 && yp < hauteur && xp >= 0 && xp < largueur)
    	                           la_ligne[x] = image1[yp][xp];
    	                    else
    	                           la_ligne[x] = 0;
    	             
    	                    numerateur_x += a11;
    	                    numerateur_y += a21;
    	                    denominateur += a31;
    	             }
    	        
    	             numerateur_x_i += a12;
    	             numerateur_y_i += a22;
    	             denominateur_i += a32;
    	               
    	       }
    }

    
	public static BufferedImage getThresholdImage(BufferedImage image) {

		// convert into a usable format
		GrayF32 input = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
		GrayU8 binary = new GrayU8(input.width, input.height);
		GThresholdImageOps.threshold(input, binary, ImageStatistics.mean(input), true);
		return VisualizeBinaryData.renderBinary(binary, false, null);
	}
	
	public static GrayU8 getThresholdBinary(BufferedImage image) {

		// convert into a usable format
		GrayF32 input = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
		GrayU8 binary = new GrayU8(input.width, input.height);
		return GThresholdImageOps.threshold(input, binary, ImageStatistics.mean(input), true);
		
	}
	
	public static <T extends ImageGray<T>, D extends ImageGray<D>> void detectLines(BufferedImage image,
			Class<T> imageType, Class<D> derivType) {
		// convert the line into a single band image
		T input = ConvertBufferedImage.convertFromSingle(image, null, imageType);

		// Comment/uncomment to try a different type of line detector
		DetectLineHoughPolar<T, D> detector = FactoryDetectLineAlgs.houghPolar(
				new ConfigHoughPolar(3, 30, 2, Math.PI / 180, edgeThreshold, maxLines), imageType, derivType);

		List<LineParametric2D_F32> found = detector.detect(input);
		
		//System.out.println(found);
		
		Graphics2D g2 = image.createGraphics();
		g2.setColor(Color.ORANGE);
		g2.setStroke(new BasicStroke(4));
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		
		// display the results
		ImageLinePanel gui = new ImageLinePanel();
		gui.setImage(image);
		gui.setLines(found);
		gui.setPreferredSize(new Dimension(image.getWidth(),image.getHeight()));
		
		lp.addItem(gui, "Found Lines");

	}
	
	
	public static<T extends ImageGray<T>, D extends ImageGray<D>>
	void detectLineSegments( BufferedImage image ,
							 Class<T> imageType ,
							 Class<D> derivType )
	{
		// convert the line into a single band image
		T input = ConvertBufferedImage.convertFromSingle(image, null, imageType );

		// Comment/uncomment to try a different type of line detector
		DetectLineSegmentsGridRansac<T,D> detector = FactoryDetectLineAlgs.lineRansac(40, 30, 2.36, true, imageType, derivType);

		List<LineSegment2D_F32> found = detector.detect(input);

		// display the results
		ImageLinePanel gui = new ImageLinePanel();
		gui.setImage(image);
		gui.setLineSegments(found);
		gui.setPreferredSize(new Dimension(image.getWidth(),image.getHeight()));

		lp.addItem(gui, "Found Line Segments");
		ShowImages.showWindow(lp,"Superpixels", true);
	}
    
	//public static boolean is
	
    public static void threshold( String imageName ) {
		BufferedImage image = UtilImageIO.loadImage(imageName);

		// convert into a usable format
		GrayF32 input = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
		GrayU8 binary = new GrayU8(input.width,input.height);

		// Display multiple images in the same window
		ListDisplayPanel gui = new ListDisplayPanel();

		// Global Methods
		GThresholdImageOps.threshold(input, binary, ImageStatistics.mean(input), true);
		gui.addImage(VisualizeBinaryData.renderBinary(binary, false, null),"Global: Mean");
		//instance.imageShower().show(ImageIO.read(new File("")), "original");
		GThresholdImageOps.threshold(input, binary, GThresholdImageOps.computeOtsu(input, 0, 255), true);
		gui.addImage(VisualizeBinaryData.renderBinary(binary, false, null),"Global: Otsu");
		GThresholdImageOps.threshold(input, binary, GThresholdImageOps.computeEntropy(input, 0, 255), true);
		gui.addImage(VisualizeBinaryData.renderBinary(binary, false, null),"Global: Entropy");


		// Show the image image for reference
		gui.addImage(ConvertBufferedImage.convertTo(input,null),"Input Image");

		String fileName =  imageName.substring(imageName.lastIndexOf('/')+1);
		ShowImages.showWindow(gui,fileName);
	}
    
    
    private static char getIndexOfLargest(float[] array, boolean isNumber) {
        if (array == null || array.length == 0) return '\0'; // null or empty
        int largest = 0;
        int n = 10;
        char c;
        int len = array.length;
        if (isNumber) {
            len = 10;
            n = 0;
        }
        for (int i = n; i < len; i++) {
            if (array[i] > array[largest])
                largest = i;
        }
        if (largest < 10) {
            c = (char) (largest + 48);
        } else {
            c = (char) (largest + 55);
        }
        return c; // position of the first largest found
    }
   
    
    public static NachImageResult label(BufferedImage image) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{
    	NachImageResult nachImageResult = new NachImageResult();
    
		//BufferedImage bankNameImage = null;
		//BufferedImage monthlyTickImage = null;
    	
    	//System.out.println("values: "+image.getHeight()+" "+image.getWidth());
    	// convert into a usable format
		GrayF32 input = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
		GrayU8 binary = new GrayU8(input.width,input.height);
		GrayS32 label = new GrayS32(input.width,input.height);

		// Select a global threshold using Otsu's method.
		double threshold = 240.0;

		// Apply the threshold to create a binary image
		ThresholdImageOps.threshold(input,binary,(float)threshold,false);
		
		GrayU8 threshholdedImage = ThresholdImageOps.threshold(input,new GrayU8(input.width,input.height),(float)threshold,true);

		// Detect blobs inside the image using an 8-connect rule
		List<Contour> contours = BinaryImageOps.contour(binary, ConnectRule.EIGHT, label);

		// colors of contours
		int colorExternal = 0xFFFFFF;
		int colorInternal = 0xFF2020;
		
		// display the results
		BufferedImage visualBinary = VisualizeBinaryData.renderBinary(binary, false, null);
		BufferedImage visualFiltered = VisualizeBinaryData.renderBinary(binary, false, null);
		BufferedImage visualLabel = rLabeledBG(label, contours.size(), null);
		BufferedImage visualContour = VisualizeBinaryData.renderContours(contours, colorExternal, colorInternal,
				input.width, input.height, null);
		BufferedImage thresholdImage = VisualizeBinaryData.renderBinary(threshholdedImage, false, null);

		gui2.addImage(visualBinary, "Binary Original");
		gui2.addImage(thresholdImage, "Thresholded Image");
		gui2.addImage(visualFiltered, "Binary Filtered");
		gui2.addImage(visualLabel, "Labeled Blobs");
		gui2.addImage(visualContour, "Contours");
		
		List<Segment> accountNumber = new ArrayList<Segment>();
		List<Segment> ifscCode = new ArrayList<Segment>();
		List<Segment> monthly = new ArrayList<Segment>();
		List<Segment> quarterly = new ArrayList<Segment>();
		List<Segment> halfYearly = new ArrayList<Segment>();
		List<Segment> yearly = new ArrayList<Segment>();
		List<Segment> reference1 = new ArrayList<Segment>();
		List<Segment> debitTick = new ArrayList<Segment>();
		List<Segment> fixedAmount = new ArrayList<Segment>();
		
		int accountNumberStartX = (int)(0.11770833333333333 * image.getWidth());
		int accountNumberStartY = (int)(0.24166666666666667 * image.getHeight());
		int accountNumberEndX =  (int)(0.9958333333333333* image.getWidth());
		int accountNumberEndY =  (int)(0.3145833333333333 * image.getHeight());
		
		int ifscCodeStartX = (int)(0.446875 * image.getWidth());
		int ifscCodeStartY = (int)(0.3125 * image.getHeight());
		int ifscCodeEndX =  (int)(0.7375 * image.getWidth());
		int ifscCodeEndY =  (int)(0.3854166666666667 * image.getHeight());

        int monthlyStartX = (int) (NachConstants.MONTHLY_START_X * image.getWidth());
        int monthlyStartY = (int) (NachConstants.MONTHLY_START_Y * image.getHeight());
        int monthlyEndX = (int) (NachConstants.MONTHLY_END_X * image.getWidth());
        int monthlyEndY = (int) (NachConstants.MONTHLY_END_Y * image.getHeight());

        int quarterlyStartX = (int) (NachConstants.QUARTERLY_START_X * image.getWidth());
        int quarterlyStartY = (int) (NachConstants.QUARTERLY_START_Y * image.getHeight());
        int quarterlyEndX = (int) (NachConstants.QUARTERLY_END_X * image.getWidth());
        int quarterlyEndY = (int) (NachConstants.QUARTERLY_END_Y * image.getHeight());

        int halfyearlyStartX = (int) (NachConstants.HALF_YEARLY_START_X * image.getWidth());
        int halfyearlyStartY = (int) (NachConstants.HALF_YEARLY_START_Y * image.getHeight());
        int halfyearlyEndX = (int) (NachConstants.HALF_YEARLY_END_X * image.getWidth());
        int halfyearlyEndY = (int) (NachConstants.HALF_YEARLY_END_X * image.getHeight());

        int yearlyStartX = (int) (NachConstants.YEARLY_START_X * image.getWidth());
        int yearlyStartY = (int) (NachConstants.YEARLY_START_Y * image.getHeight());
        int yearlyEndX = (int) (NachConstants.YEARLY_END_X * image.getWidth());
        int yearlyEndY = (int) (NachConstants.YEARLY_END_Y * image.getHeight());

        int fixedAmountStartX = (int) (NachConstants.FIXED_AMOUNT_START_X * image.getWidth());
        int fixedAmountStartY = (int) (NachConstants.FIXED_AMOUNT_START_Y * image.getHeight());
        int fixedAmountEndX = (int) (NachConstants.FIXED_AMOUNT_END_X * image.getWidth());
        int fixedAmountEndY = (int) (NachConstants.FIXED_AMOUNT_END_Y * image.getHeight());
		
        int bankNameStartX = (int) (NachConstants.BANK_NAME_START_X * image.getWidth());
        int bankNameStartY = (int) (NachConstants.BANK_NAME_START_Y * image.getHeight());
        int bankNameEndX = (int) (NachConstants.BANK_NAME_END_X * image.getWidth());
        int bankNameEndY = (int) (NachConstants.BANK_NAME_END_Y * image.getHeight());
		
        int reference1StartX = (int)(0.093125 * image.getWidth());
		int reference1StartY = (int)(0.5 * image.getHeight());
		int reference1EndX =  (int)(0.62125 * image.getWidth());
		int reference1EndY =  (int)(0.59875 * image.getHeight());
        
//        int reference1StartX = (int) (NachConstants.REFERENCE_CODE_START_X * image.getWidth());
//        int reference1StartY = (int) (NachConstants.REFERENCE_CODE_START_Y * image.getHeight());
//        int reference1EndX = (int) (NachConstants.REFERENCE_CODE_END_X * image.getWidth());
//        int reference1EndY = (int) (NachConstants.REFERENCE_CODE_END_Y * image.getHeight());
	    
	    int signatureStartX = (int) (NachConstants.SIGNATURE_START_X * image.getWidth());
	    int signatureStartY = (int) (NachConstants.SIGNATURE_START_Y * image.getHeight());
	    
	    int nameStartX = (int) (NachConstants.NAME_START_X * image.getWidth());
	    int nameStartY = (int) (NachConstants.NAME_START_Y * image.getHeight());
	    
	    int debitStartX = (int) (NachConstants.DEBIT_START_X * image.getWidth());
	    int debitStartY = (int) (NachConstants.DEBIT_START_Y * image.getHeight());
	    int debitEndX = (int) (NachConstants.DEBIT_END_X * image.getWidth());
	    int debitEndY = (int) (NachConstants.DEBIT_END_Y * image.getHeight());
	    
	    
		
		for(Segment s: list2){
			int x = s.getX();
			int y = s.getY();
			int wd = s.getWidth();
			int ht = s.getHeight();

			if(x>accountNumberStartX && y> accountNumberStartY && (x+wd)< accountNumberEndX  && (y+ht)<accountNumberEndY && wd>15 && ht>15){
				if(s.width > 20 && s.height > 25) {
					accountNumber.add(s);
				}
			}
			
			if(x>ifscCodeStartX && y>ifscCodeStartY && (x+wd)<ifscCodeEndX && (y+ht)<ifscCodeEndY && wd>15 && ht>15){
				if(s.width > 20 && s.height > 25) {
					ifscCode.add(s);
				}
			}
			
			if(x>monthlyStartX && y>monthlyStartY && (x+wd)<monthlyEndX && (y+ht)<monthlyEndY){
				monthly.add(s);
			}
			
			if(x>quarterlyStartX && y>quarterlyStartY && (x+wd)<quarterlyEndX && (y+ht)<quarterlyEndY){
				quarterly.add(s);
			}
			
			if(x>halfyearlyStartX && y>halfyearlyStartY && (x+wd)<halfyearlyEndX && (y+ht)<halfyearlyEndY){
				halfYearly.add(s);
			}
			
			if(x>yearlyStartX && y>yearlyStartY && (x+wd)<yearlyEndX && (y+ht)<yearlyEndY){
				yearly.add(s);
			}
			
			if(x>fixedAmountStartX && y>fixedAmountStartY && (x+wd)<fixedAmountEndX && (y+ht)<fixedAmountEndY){
				fixedAmount.add(s);
			}
		}
		
		Collections.sort(accountNumber, new SegmentComparator());
		Collections.sort(ifscCode, new SegmentComparator());
		
		Map<NachField, List<Segment>> map = new HashMap<>();
		
        map.put(NachField.ACCOUNT_NUMBER, accountNumber);
        map.put(NachField.IFSC_CODE, ifscCode);
        map.put(NachField.MONTHLY, monthly);
        map.put(NachField.QUARTERLY, quarterly);
        map.put(NachField.HALF_YEARLY, halfYearly);
        map.put(NachField.YEARLY, yearly);
        map.put(NachField.FIXED_AMOUNT, fixedAmount);
        
        StringBuilder accountNumberString = new StringBuilder();
		for(Segment p: accountNumber){
			buff = thresholdImage.getSubimage(p.getX(), p.getY(), p.getWidth(), p.getHeight());
			char val = detectCharacter(buff, true);
			if( val != '\0') {
				accountNumberString.append(val);		
			}
		}
		//accountNumberString.replace(null, "");
		nachImageResult.setAccountNumber(accountNumberString.toString());
		
		
		StringBuilder ifscCodeString = new StringBuilder();
		int i = 0;
		for(Segment p: ifscCode){
			i++;
			buff = thresholdImage.getSubimage(p.getX(), p.getY(), p.getWidth(), p.getHeight());
			boolean isNumber = (i < 5) ? false : true;
			char val = detectCharacter(buff, isNumber);
			if( val != '\0') {
				ifscCodeString.append(val);		
			}		
		}
		
		nachImageResult.setIfscCode(ifscCodeString.toString());
		nachImageResult.setMonthlyTicked(checkboxTicked(monthly, thresholdImage));
		nachImageResult.setQuarterlyTicked(checkboxTicked(quarterly, thresholdImage));
		nachImageResult.setHalfYearlyTicked(checkboxTicked(quarterly, thresholdImage));
		nachImageResult.setYearlyTicked(checkboxTicked(yearly, thresholdImage));
		nachImageResult.setFixedAmountTicked(checkboxTicked(fixedAmount, thresholdImage));
	
		BufferedImage signatureImage = thresholdImage.getSubimage(signatureStartX, signatureStartY, 344, 139);
		BufferedImage debitTickImage = thresholdImage.getSubimage(debitStartX, debitStartY, debitEndX-debitStartX, debitEndY-debitStartY);
		BufferedImage bankNameImage = thresholdImage.getSubimage(bankNameStartX, bankNameStartY, bankNameEndX-bankNameStartX, bankNameEndY-bankNameStartY);
		BufferedImage nameImage = thresholdImage.getSubimage(nameStartX, nameStartY, 333, 59);
		BufferedImage reference = thresholdImage.getSubimage(reference1StartX, reference1StartY, reference1EndX-reference1StartX, reference1EndY-reference1StartY);

		nachImageResult.setFormNumber(OCR(reference));
		nachImageResult.setDebitTicked(isDebitTickPresent(OCR(debitTickImage)));
		
		nachImageResult.setBankNamePresent(bankNameExists(bankNameImage));
		nachImageResult.setSignaturePresent(signatureExists(signatureImage));
		nachImageResult.setNamePresent(nameExists(nameImage));
		
		System.out.println(nachImageResult.toString());
		return nachImageResult;
	}
    
    
    private static boolean isDebitTickPresent(String str) {
    	System.out.println("Debit string" + str);
	    if (str == null) 
	    	return true;
	    
        if (str.startsWith("SB/CA")) 
            return false;
        
        return true;
  	}
    
    public static BufferedImage rLabeledBG(GrayS32 labelImage, int numRegions, BufferedImage out) {

		int colors[] = new int[numRegions+1];

		Random rand = new Random(123);
		for( int i = 0; i < colors.length; i++ ) {
			colors[i] = rand.nextInt();
		}
		colors[0] = 0;

		return rLabeled(labelImage, colors, out);
	}
    
    public static BufferedImage rLabeled(GrayS32 labelImage, int colors[], BufferedImage out) {

		if( out == null ) {
			out = new BufferedImage(labelImage.getWidth(),labelImage.getHeight(),BufferedImage.TYPE_INT_RGB);
		}
		_renderLabeled(labelImage, out, colors);
		return out;
	}
    
    private static void _renderLabeled(GrayS32 labelImage, BufferedImage out, int[] colors) {
		int w = labelImage.getWidth();
		int h = labelImage.getHeight();
		
		Map<Integer,TreeSet<Integer>> valuesX = new HashMap<Integer,TreeSet<Integer>>();
		Map<Integer,TreeSet<Integer>> valuesY = new HashMap<Integer,TreeSet<Integer>>();
		
		int[] min_X = new int[colors.length];
		int[] max_X = new int[colors.length];
		int[] min_Y = new int[colors.length];
		int[] max_Y = new int[colors.length];	
		
		for(int i=0;i<labelImage.data.length;i++){
			int x = i%w ;
			int y = i/w;
			
			
			if(min_X[labelImage.data[i]]==0){
				min_X[labelImage.data[i]]=x;
			}else{
				if(x<min_X[labelImage.data[i]]){
					min_X[labelImage.data[i]]=x;
				}
			}
			
			if(max_X[labelImage.data[i]]==0){
				max_X[labelImage.data[i]]=x;
			}else{
				if(x>max_X[labelImage.data[i]]){
					max_X[labelImage.data[i]]=x;
				}
			}
			
			if(min_Y[labelImage.data[i]]==0){
				min_Y[labelImage.data[i]]=y;
			}else{
				if(y<min_Y[labelImage.data[i]]){
					min_Y[labelImage.data[i]]=y;
				}
			}
			
			if(max_Y[labelImage.data[i]]==0){
				max_Y[labelImage.data[i]]=y;
			}else{
				if(y>max_Y[labelImage.data[i]]){
					max_Y[labelImage.data[i]]=y;
				}
			}
			
			if(!valuesX.containsKey(labelImage.data[i])){
				TreeSet<Integer> set = new TreeSet<>();
				set.add(x);
				valuesX.put(labelImage.data[i], set);
			}else{
				TreeSet<Integer> set = valuesX.get(labelImage.data[i]);
				set.add(x);
				valuesX.put(labelImage.data[i], set);
			}
			
			
			if(!valuesY.containsKey(labelImage.data[i])){
				TreeSet<Integer> set = new TreeSet<>();
				set.add(y);
				valuesY.put(labelImage.data[i], set);
			}else{
				TreeSet<Integer> set = valuesY.get(labelImage.data[i]);
				set.add(y);
				valuesY.put(labelImage.data[i], set);
			}		
		}
		
		for(int i=0;i<colors.length;i++){
			
			int minX = min_X[i];
			int maxX = max_X[i];
			
			int minY = min_Y[i];
			int maxY = max_Y[i];
			
			Segment segment = new Segment(minX, minY, maxX-minX, maxY-minY);
			
			list2.add(segment);
			
			
		}
				
		Set<Integer> set = new HashSet<Integer>();
		Set<Integer> set1 = new HashSet<Integer>();

		for( int y = 0; y < h; y++ ) {
			int indexSrc = labelImage.startIndex + y*labelImage.stride;
			for( int x = 0; x < w; x++ ) {
				int val = labelImage.data[indexSrc++];
				int rgb = colors[val];
				set.add(rgb);
				set1.add(val);
				out.setRGB(x,y,rgb);	
			}
		}		
	}
    
	private static char detectCharacter(BufferedImage bufferedImage, boolean isNumber) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        float[][][][] colorArr = new float[1][1][28][28];
		File file = new File("C:\\Users\\nayaninternkaleidofi\\Downloads\\combined_model_2.h5");
		MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(file.getAbsolutePath(), true);
        BufferedImage scaled = getResizedImage(bufferedImage, 28, 28);
		boolean isBlankSegment = isBlank(scaled,colorArr);
        if (!isBlankSegment) {
            INDArray imageToNDArray = Nd4j.create(colorArr);	
    		INDArray output = model.output(imageToNDArray);
    		float[] mResult = output.toFloatVector();
            return getIndexOfLargest(mResult, isNumber);
            
        }
        return '\0';
    }
    
    private static boolean isBlank(BufferedImage bufferedImage, float[][][][] colorArr) {
        float sum = 0;
        for (int i = 0; i < bufferedImage.getWidth(); ++i) {
            for (int j = 0; j < bufferedImage.getHeight(); ++j) {
            	Color color = new Color(bufferedImage.getRGB(j, i));
	            int red = color.getRed();
                colorArr[0][0][i][j] = (float) red / (float) 255.0;
                sum = sum + colorArr[0][0][i][j];
            }
        }
        float density_percent = (sum * 100.0f / (float) (NachConstants.SEGMENT_HEIGHT * NachConstants.SEGMENT_WIDTH));
        return (density_percent < 10.0f);
    }
    
	public static List<Point2D_F64> fitBinaryImage(GrayF32 input) {

		GrayU8 binary = new GrayU8(input.width,input.height);
		BufferedImage polygon = new BufferedImage(input.width,input.height,BufferedImage.TYPE_INT_RGB);

		// the mean pixel value is often a reasonable threshold when creating a binary image
		double mean = ImageStatistics.mean(input);

		// create a binary image by thresholding
		ThresholdImageOps.threshold(input, binary, (float) mean, true);

		// reduce noise with some filtering
		GrayU8 filtered = BinaryImageOps.erode8(binary, 1, null);
		filtered = BinaryImageOps.dilate8(filtered, 1, null);

		// Find internal and external contour around each shape
		List<Contour> contours = BinaryImageOps.contour(filtered, ConnectRule.EIGHT,null);

		// Fit a polygon to each shape and draw the results
		Graphics2D g2 = polygon.createGraphics();
		g2.setStroke(new BasicStroke(2));
		
		int tl_x=0;
		int tl_y=0;
		
		int br_x = input.width;
		int br_y = input.height;
		
		int tr_x=0;
		int tr_y=input.height;
		
		int bl_x=input.width;
		int bl_y=0;
		
		boolean found=false;

		for( Contour c : contours ) {
			// Fit the polygon to the found external contour.  Note loop = true
			List<PointIndex_I32> vertexes = ShapeFittingOps.fitPolygon(c.external,true, minSide,cornerPenalty);
//
			g2.setColor(Color.RED);
			VisualizeShapes.drawPolygon(vertexes,true,g2);

			// handle internal contours now
			g2.setColor(Color.BLUE);
			for( List<Point2D_I32> internal : c.internal ) {
				 vertexes = ShapeFittingOps.fitPolygon(internal,true, minSide,cornerPenalty);
				if(vertexes.size()==4){		
					
					int ax = vertexes.get(0).getX();
					int ay = vertexes.get(0).getY();
					int bx = vertexes.get(1).getX();
					int by = vertexes.get(1).getY();
					int cx = vertexes.get(2).getX();
					int cy = vertexes.get(2).getY();
					int dx = vertexes.get(3).getX();
					int dy = vertexes.get(3).getY();
					
					int sum_a = ax+ay;
					int sum_b = bx+by;
					int sum_c = cx+cy;
					int sum_d = dx+dy;
					
					int max_sum = Math.max(Math.max(sum_a, sum_b), Math.max(sum_c, sum_d));
					
					if(max_sum==sum_a){
						br_x = ax;
						br_y = ay;
						ax=-1;
						ay =-1;
					}else if(max_sum==sum_b){
						br_x = bx;
						br_y = by;
						bx =-1;
						by =-1;
					}else if(max_sum==sum_c){
						br_x= cx;
						br_y = cy;
						cx= -1;
						cy=-1;
					}else{
						br_x = dx;
						br_y = dy;
						dx=-1;
						dy=-1;
					}
					
					int min_sum = Math.min(Math.min(sum_a, sum_b), Math.min(sum_c, sum_d));		
			
					if(min_sum==sum_a){
						tl_x = ax;
						tl_y = ay;
						ax=-1;
						ay =-1;
					}else if(min_sum==sum_b){
						tl_x = bx;
						tl_y = by;
						bx=-1;
						by =-1;
					}else if(min_sum==sum_c){
						tl_x = cx;
						tl_y = cy;
						cx=-1;
						cy =-1;
					}else{
						tl_x = dx;
						tl_y = dy;
						dx=-1;
						dy =-1;
					}
					
					int diff_a=-1;
					if(ax!=-1 && ay!=-1){
						diff_a = Math.abs(ax-ay);
					}	
					int diff_b = -1;
					if(bx!=-1 &&  by!=-1){
						diff_b = Math.abs(bx-by);
					}
					int diff_c = -1;
					if(cx!=-1 &&  cy!=-1){
						diff_c = Math.abs(cx-cy);
					}
					int diff_d = -1;
					if(dx!=-1 &&  dy!=-1){
						diff_d = Math.abs(dx-dy);
					}
					
					int max_diff = Math.max(Math.max(diff_a, diff_b), Math.max(diff_c, diff_d));
					
					if(max_diff==diff_a){
						tr_x = ax;
						tr_y = ay;
						diff_a=-1;
					}else if(max_diff==diff_b){
						tr_x = bx;
						tr_y = by;
						diff_b=-1;
					}else if(max_diff==diff_c){
						tr_x = cx;
						tr_y = cy;
						diff_c=-1;
					}else{
						tr_x = dx;
						tr_y = dy;
						diff_d=-1;
					}						
					
					if(diff_a!=-1){
						bl_x = ax;
						bl_y = ay;
					}else if(diff_b!=-1){
						bl_x = bx;
						bl_y = by;
					}else if(diff_c!=-1){
						bl_x = cx;
						bl_y = cy;
					}else{
						bl_x = dx;
						bl_y = dy;
					}			
					
					
					int wd = br_x - tl_x;
					int ht = br_y - tl_y;
					
					
					double wd_percent = ((double)wd/(double)input.width)*100;
					double ht_percent = ((double)ht/(double)input.height)*100;				
					
									
					if(wd_percent>55.00 && wd_percent<98.00 && ht_percent>55.00 && ht_percent<98.00){
						found=true;
						VisualizeShapes.drawPolygon(vertexes,true,g2);
						break;
					}
				}
				//VisualizeShapes.drawPolygon(vertexes,true,g2);
			}
			if(found){
				break;
			}
		}
		List<Point2D_F64> points = new ArrayList<Point2D_F64>();
		
		if(found){
			points.add(new Point2D_F64((double)tl_x, (double)tl_y));
			points.add(new Point2D_F64((double)tr_x, (double)tr_y));
			points.add(new Point2D_F64((double)br_x, (double)br_y));
			points.add(new Point2D_F64((double)bl_x, (double)bl_y));
			//System.out.println("OP: "+points);
		}

		//System.out.println(tl_x+" "+tl_y+ " "+ " "+tr_x+" "+ tr_y);
		//gui2.addImage(polygon, "Binary Blob Contours");
		return points;
	}
	/**
	 * Fits a sequence of line-segments into a sequence of points found using the Canny edge detector.  In this case
	 * the points are not connected in a loop. The canny detector produces a more complex tree and the fitted
	 * points can be a bit noisy compared to the others.
	 */
	public static void fitCannyEdges( GrayF32 input ) {

		BufferedImage displayImage = new BufferedImage(input.width,input.height,BufferedImage.TYPE_INT_RGB);

		// Finds edges inside the image
		CannyEdge<GrayF32,GrayF32> canny =
				FactoryEdgeDetectors.canny(2, true, true, GrayF32.class, GrayF32.class);

		canny.process(input,0.1f,0.3f,null);
		List<EdgeContour> contours = canny.getContours();

		Graphics2D g2 = displayImage.createGraphics();
		g2.setStroke(new BasicStroke(2));

		// used to select colors for each line
		Random rand = new Random(234);

		for( EdgeContour e : contours ) {
			g2.setColor(new Color(rand.nextInt()));

			for(EdgeSegment s : e.segments ) {
				// fit line segments to the point sequence.  Note that loop is false
				List<PointIndex_I32> vertexes = ShapeFittingOps.fitPolygon(s.points,false, minSide,cornerPenalty);

				VisualizeShapes.drawPolygon(vertexes, false, g2);
			}
		}

		//gui2.addImage(displayImage, "Canny Trace");
	}
	
	public static List<Point2D_F64> contourExternal(GrayU8 input, ConnectRule rule ) {
		BinaryContourFinder alg = FactoryBinaryContourFinder.linearExternal();
		alg.setConnectRule(rule);
		alg.process(input);

		return convertContours(alg,input);
	}
	
	public static List<Point2D_F64> convertContours(BinaryContourInterface alg ,GrayU8 input) {

		List<ContourPacked> contours = alg.getContours();
		BufferedImage displayImage = new BufferedImage(input.width,input.height,BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = displayImage.createGraphics();
		g2.setStroke(new BasicStroke(2));

		// used to select colors for each line
		Random rand = new Random(234);
		
		int tl_x=0;
		int tl_y=0;
		
		int br_x = input.width;
		int br_y = input.height;
		
		int tr_x=0;
		int tr_y=input.height;
		
		boolean found=false;
		
		int bl_x=input.width;
		int bl_y=0;

		for (int i = 0; i < contours.size(); i++) {
			ContourPacked p = contours.get(i);
			Contour c = new Contour();
			c.external = BinaryContourInterface.copyContour(alg,p.externalIndex);
			List<PointIndex_I32> vertexes = ShapeFittingOps.fitPolygon(c.external,true, minSide,cornerPenalty);
			g2.setColor(new Color(rand.nextInt()));
			if(vertexes.size()==4){		
				//System.out.println("Vertexes: "+vertexes);
				int ax = vertexes.get(0).getX();
				int ay = vertexes.get(0).getY();
				int bx = vertexes.get(1).getX();
				int by = vertexes.get(1).getY();
				int cx = vertexes.get(2).getX();
				int cy = vertexes.get(2).getY();
				int dx = vertexes.get(3).getX();
				int dy = vertexes.get(3).getY();	
				
				int sum_a = ax+ay;
				int sum_b = bx+by;
				int sum_c = cx+cy;
				int sum_d = dx+dy;
				
				int max_sum = Math.max(Math.max(sum_a, sum_b), Math.max(sum_c, sum_d));
				
				if(max_sum==sum_a){
					br_x = ax;
					br_y = ay;
					ax=-1;
					ay =-1;
				}else if(max_sum==sum_b){
					br_x = bx;
					br_y = by;
					bx =-1;
					by =-1;
				}else if(max_sum==sum_c){
					br_x= cx;
					br_y = cy;
					cx= -1;
					cy=-1;
				}else{
					br_x = dx;
					br_y = dy;
					dx=-1;
					dy=-1;
				}
				
				int min_sum = Math.min(Math.min(sum_a, sum_b), Math.min(sum_c, sum_d));		
		
				if(min_sum==sum_a){
					tl_x = ax;
					tl_y = ay;
					ax=-1;
					ay =-1;
				}else if(min_sum==sum_b){
					tl_x = bx;
					tl_y = by;
					bx=-1;
					by =-1;
				}else if(min_sum==sum_c){
					tl_x = cx;
					tl_y = cy;
					cx=-1;
					cy =-1;
				}else{
					tl_x = dx;
					tl_y = dy;
					dx=-1;
					dy =-1;
				}
				
				int diff_a=-1;
				if(ax!=-1 && ay!=-1){
					diff_a = Math.abs(ax-ay);
				}	
				int diff_b = -1;
				if(bx!=-1 &&  by!=-1){
					diff_b = Math.abs(bx-by);
				}
				int diff_c = -1;
				if(cx!=-1 &&  cy!=-1){
					diff_c = Math.abs(cx-cy);
				}
				int diff_d = -1;
				if(dx!=-1 &&  dy!=-1){
					diff_d = Math.abs(dx-dy);
				}
				
				int max_diff = Math.max(Math.max(diff_a, diff_b), Math.max(diff_c, diff_d));
				
				if(max_diff==diff_a){
					tr_x = ax;
					tr_y = ay;
					diff_a=-1;
				}else if(max_diff==diff_b){
					tr_x = bx;
					tr_y = by;
					diff_b=-1;
				}else if(max_diff==diff_c){
					tr_x = cx;
					tr_y = cy;
					diff_c=-1;
				}else{
					tr_x = dx;
					tr_y = dy;
					diff_d=-1;
				}						
				
				if(diff_a!=-1){
					bl_x = ax;
					bl_y = ay;
				}else if(diff_b!=-1){
					bl_x = bx;
					bl_y = by;
				}else if(diff_c!=-1){
					bl_x = cx;
					bl_y = cy;
				}else{
					bl_x = dx;
					bl_y = dy;
				}
				
				int wd = br_x - tl_x;
				int ht = br_y - tl_y;
				
				
				
				
				double wd_percent = ((double)wd/(double)input.width)*100;
				double ht_percent = ((double)ht/(double)input.height)*100;
				
				
								
				if(wd_percent>55.00 && wd_percent<98.00 && ht_percent>55.00 && ht_percent<98.00){
					found=true;
					VisualizeShapes.drawPolygon(vertexes,true,g2);
					break;
				}
			}
		}
		
		List<Point2D_F64> points = new ArrayList<Point2D_F64>();
		
		if(found){
			points.add(new Point2D_F64((double)tl_x, (double)tl_y));
			points.add(new Point2D_F64((double)tr_x, (double)tr_y));
			points.add(new Point2D_F64((double)br_x, (double)br_y));
			points.add(new Point2D_F64((double)bl_x, (double)bl_y));
		}
		
		return points;
	}

	/**
	 * Detects contours inside the binary image generated by canny.  Only the external contour is relevant. Often
	 * easier to deal with than working with Canny edges directly.
	 * @return 
	 */
	public static List<Point2D_F64> fitCannyBinary( GrayF32 input, BufferedImage src ) {

		
		// System.out.println("data "+Arrays.toString(input.data));

		
		BufferedImage displayImage = new BufferedImage(input.width,input.height,BufferedImage.TYPE_INT_RGB);
		GrayU8 binary = new GrayU8(input.width,input.height);

		double threshold = GThresholdImageOps.computeOtsu(input, 0, 255);

		// create a binary image by thresholding
		ThresholdImageOps.threshold(input, binary, (float) threshold, true);
		
		// reduce noise with some filtering
		GrayU8 filtered = BinaryImageOps.erode8(binary, 1, null);
		filtered = BinaryImageOps.dilate8(filtered, 1, null);
		
		//System.out.println("Detection "+contourExternal(binary, ConnectRule.EIGHT));

		List<Contour> contours = BinaryImageOps.contourExternal(binny, ConnectRule.EIGHT);
		
		
		//System.out.println(contours);
		
		Graphics2D g2 = displayImage.createGraphics();
		g2.setStroke(new BasicStroke(2));

		// used to select colors for each line
		Random rand = new Random(234);
		
		int tl_x=0;
		int tl_y=0;
		
		int br_x = input.width;
		int br_y = input.height;
		
		int tr_x=0;
		int tr_y=input.height;
		
		int bl_x=input.width;
		int bl_y=0;
		
		boolean found=false;

		for( Contour c : contours ) {
			List<PointIndex_I32> vertexes = ShapeFittingOps.fitPolygon(c.external,true, minSide,cornerPenalty);
			g2.setColor(new Color(rand.nextInt()));
			
			if(vertexes.size()==4){		
				//System.out.println("Vertexes: "+vertexes);
				int ax = vertexes.get(0).getX();
				int ay = vertexes.get(0).getY();
				int bx = vertexes.get(1).getX();
				int by = vertexes.get(1).getY();
				int cx = vertexes.get(2).getX();
				int cy = vertexes.get(2).getY();
				int dx = vertexes.get(3).getX();
				int dy = vertexes.get(3).getY();	
				
				int sum_a = ax+ay;
				int sum_b = bx+by;
				int sum_c = cx+cy;
				int sum_d = dx+dy;
				
				int max_sum = Math.max(Math.max(sum_a, sum_b), Math.max(sum_c, sum_d));
				
				if(max_sum==sum_a){
					br_x = ax;
					br_y = ay;
					ax=-1;
					ay =-1;
				}else if(max_sum==sum_b){
					br_x = bx;
					br_y = by;
					bx =-1;
					by =-1;
				}else if(max_sum==sum_c){
					br_x= cx;
					br_y = cy;
					cx= -1;
					cy=-1;
				}else{
					br_x = dx;
					br_y = dy;
					dx=-1;
					dy=-1;
				}
				
				int min_sum = Math.min(Math.min(sum_a, sum_b), Math.min(sum_c, sum_d));		
		
				if(min_sum==sum_a){
					tl_x = ax;
					tl_y = ay;
					ax=-1;
					ay =-1;
				}else if(min_sum==sum_b){
					tl_x = bx;
					tl_y = by;
					bx=-1;
					by =-1;
				}else if(min_sum==sum_c){
					tl_x = cx;
					tl_y = cy;
					cx=-1;
					cy =-1;
				}else{
					tl_x = dx;
					tl_y = dy;
					dx=-1;
					dy =-1;
				}
				
				int diff_a=-1;
				if(ax!=-1 && ay!=-1){
					diff_a = Math.abs(ax-ay);
				}	
				int diff_b = -1;
				if(bx!=-1 &&  by!=-1){
					diff_b = Math.abs(bx-by);
				}
				int diff_c = -1;
				if(cx!=-1 &&  cy!=-1){
					diff_c = Math.abs(cx-cy);
				}
				int diff_d = -1;
				if(dx!=-1 &&  dy!=-1){
					diff_d = Math.abs(dx-dy);
				}
				
				int max_diff = Math.max(Math.max(diff_a, diff_b), Math.max(diff_c, diff_d));
				
				if(max_diff==diff_a){
					tr_x = ax;
					tr_y = ay;
					diff_a=-1;
				}else if(max_diff==diff_b){
					tr_x = bx;
					tr_y = by;
					diff_b=-1;
				}else if(max_diff==diff_c){
					tr_x = cx;
					tr_y = cy;
					diff_c=-1;
				}else{
					tr_x = dx;
					tr_y = dy;
					diff_d=-1;
				}						
				
				if(diff_a!=-1){
					bl_x = ax;
					bl_y = ay;
				}else if(diff_b!=-1){
					bl_x = bx;
					bl_y = by;
				}else if(diff_c!=-1){
					bl_x = cx;
					bl_y = cy;
				}else{
					bl_x = dx;
					bl_y = dy;
				}
				
				int wd = br_x - tl_x;
				int ht = br_y - tl_y;
				
				double wd_percent = ((double)wd/(double)input.width)*100;
				double ht_percent = ((double)ht/(double)input.height)*100;
								
				if(wd_percent>55.00 && wd_percent<98.00 && ht_percent>55.00 && ht_percent<98.00){
					found=true;
					VisualizeShapes.drawPolygon(vertexes,true,g2);
					break;
				}
			}
			//VisualizeShapes.drawPolygon(vertexes,true,g2);
		}
		List<Point2D_F64> points = new ArrayList<Point2D_F64>();
		
		if(found){
			points.add(new Point2D_F64((double)tl_x, (double)tl_y));
			points.add(new Point2D_F64((double)tr_x, (double)tr_y));
			points.add(new Point2D_F64((double)br_x, (double)br_y));
			points.add(new Point2D_F64((double)bl_x, (double)bl_y));
		}

		gui2.addImage(displayImage, "Canny Contour");
		
		return points;
	}
	
	public static BufferedImage getResizedImage(BufferedImage original, int width, int height){
		BufferedImage resized = new BufferedImage(width, height, original.getType());
		Graphics2D g = resized.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
		    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(original, 0, 0, width, height, 0, 0, original.getWidth(),
		    original.getHeight(), null);
		g.dispose();
		return resized;
	}
	
	public static List<Point2D_F64> fitBinaryImage1(GrayF32 input) {

		GrayU8 binary = new GrayU8(input.width,input.height);
		BufferedImage polygon = new BufferedImage(input.width,input.height,BufferedImage.TYPE_INT_RGB);

		// the mean pixel value is often a reasonable threshold when creating a binary image
		double mean = ImageStatistics.mean(input);

		// create a binary image by thresholding
		ThresholdImageOps.threshold(input, binary, (float) mean, true);

		// reduce noise with some filtering
		GrayU8 filtered = BinaryImageOps.erode8(binary, 1, null);
		filtered = BinaryImageOps.dilate8(filtered, 1, null);

		// Find internal and external contour around each shape
		List<Contour> contours = BinaryImageOps.contour(filtered, ConnectRule.EIGHT,null);

		// Fit a polygon to each shape and draw the results
		Graphics2D g2 = polygon.createGraphics();
		g2.setStroke(new BasicStroke(2));

		for( Contour c : contours ) {
			// Fit the polygon to the found external contour.  Note loop = true
			List<PointIndex_I32> vertexes = ShapeFittingOps.fitPolygon(c.external,true, minSide,cornerPenalty);

			g2.setColor(Color.RED);
			VisualizeShapes.drawPolygon(vertexes,true,g2);

			// handle internal contours now
			g2.setColor(Color.BLUE);
			for( List<Point2D_I32> internal : c.internal ) {
				vertexes = ShapeFittingOps.fitPolygon(internal,true, minSide,cornerPenalty);
				VisualizeShapes.drawPolygon(vertexes,true,g2);
			}
		}
		List<Point2D_F64> points = new ArrayList<Point2D_F64>();

		//System.out.println(tl_x+" "+tl_y+ " "+ " "+tr_x+" "+ tr_y);
		gui2.addImage(polygon, "Binary Blob Contours");
		return points;
	}
	
	public static String OCR(BufferedImage src) throws IOException {
		if(src == null || src.getWidth()*src.getHeight() == 0) {
			return null;
		}
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(src, "jpg", baos);
		byte[] srcData = baos.toByteArray();
		//((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		QuickstartSample sample = new QuickstartSample();
		String detectedText = sample.detectText(srcData);
		//System.out.println("Reference Code: in function" + detectedText);
		return detectedText;
		
	}
	
	public static void cannyEdge(){
//		String imageFilename = "/home/maruthi/Desktop/test_boofcv/aadhar/9.jpg";
//		BufferedImage image = UtilImageIO.loadImage(UtilIO.pathExample("/home/maruthi/Desktop/test_boofcv/aadhar/34.jpg"));

//		GrayU8 gray = ConvertBufferedImage.convertFrom(image,(GrayU8)null);
//		GrayU8 edgeImage = gray.createSameShape();

		// Create a canny edge detector which will dynamically compute the threshold based on maximum edge intensity
		// It has also been configured to save the trace as a graph.  This is the graph created while performing
		// hysteresis thresholding.
		CannyEdge<GrayU8,GrayS16> canny = FactoryEdgeDetectors.canny(2,true, true, GrayU8.class, GrayS16.class);

		// The edge image is actually an optional parameter.  If you don't need it just pass in null
//		canny.process(gray,0.1f,0.3f,edgeImage);

		// First get the contour created by canny
		List<EdgeContour> edgeContours = canny.getContours();
		// display the results
//		BufferedImage visualBinary = VisualizeBinaryData.renderBinary(edgeImage, false, null);
//		BufferedImage visualCannyContour = VisualizeBinaryData.renderContours(edgeContours,null,
//				gray.width,gray.height,null);
//		BufferedImage visualEdgeContour = new BufferedImage(gray.width, gray.height,BufferedImage.TYPE_INT_RGB);
//		VisualizeBinaryData.render(contours, (int[]) null, visualEdgeContour);
//		VisualizeBinaryData.renderContours(contours, colorInternal, colorInternal, width, height, out);
		ListDisplayPanel panel = new ListDisplayPanel();
//		panel.addImage(visualBinary,"Binary Edges from Canny");
//		panel.addImage(visualCannyContour, "Canny Trace Graph");
//		panel.addImage(visualEdgeContour,"Contour from Canny Binary");
		ShowImages.showWindow(panel,"Canny Edge", true);
		
//      detectLines(UtilImageIO.loadImage(UtilIO.pathExample(imageFilename)), GrayU8.class, GrayS16.class);
//      detectLineSegments(UtilImageIO.loadImage(UtilIO.pathExample(imageFilename)), GrayF32.class, GrayF32.class);
		
	}
	
	
	private static float getPixelDensity(BufferedImage src) {
        //BufferedImage bitmap = invert(src);
        float sum = 0.0f;
        
        for (int i = 0; i < src.getWidth(); i++) {
            for (int j = 0; j < src.getHeight(); j++) {
            	Color pixel = new Color(src.getRGB(i, j));                
                int red = pixel.getRed();
                sum = sum + (float) red / (float) 255.0;
            }
        }
        return (sum * 100.0f / (float) (src.getHeight() * src.getWidth()));
    }
	
	private static boolean bankNameExists(BufferedImage src) {
        return (getPixelDensity(src) > 10.00f);
    }


    private static boolean nameExists(BufferedImage src) {
        return (getPixelDensity(src) > 5.00f);
    }

    // Lower Threshold Because Signature written may
    private static boolean signatureExists(BufferedImage src) {
        //System.out.println("sign " + getPixelDensity(src));
        return (getPixelDensity(src) > 2.00f);
    }
	
    private static boolean checkboxTicked(List<Segment> list, BufferedImage src) {
        //System.out.println(list.size() + "hello");
        if (list.size() == 0) {
            // Unable to find Segments
            return false;
        } else if (list.size() == 1) {
            //There may be a small tick that does not intersect with the borders
        	Segment s = list.get(0);
            BufferedImage bufferedImage = src.getSubimage(s.getX(), s.getY(), s.getWidth(), s.getHeight());
            return (getPixelDensity(bufferedImage) > 20.0f);
        }
        // If there is more than 1 segment, then there is definitely something on the box that splits it
        return true;
    }
	
	
	public static NachImageResult preprocessing(BufferedImage image) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    	boolean rectangleDetected = true;
    	list2= new ArrayList<Segment>();
		
		BufferedImage th  = getThresholdImage(image);
		binny =getThresholdBinary(image);
		
		GrayF32 input = ConvertBufferedImage.convertFromSingle(th, null, GrayF32.class);

		gui2.addImage(image,"Original");
		gui2.addImage(th, "Threshold Image");
		
		List<Point2D_F64> points1 =fitCannyBinary(input,image);
		List<Point2D_F64> points =fitBinaryImage(input);
//		System.out.println(points);
//		System.out.println(points1);
		NachImageResult nachImageResult = new NachImageResult();
		
		if(points1.size()==4){
		
		 Planar<GrayF32> planar = ConvertBufferedImage.convertFromPlanar(image, null, true, GrayF32.class);

		 //gui2.addImage(ConvertBufferedImage.convertTo_F32(planar,null,true), "test");
		RemovePerspectiveDistortion<Planar<GrayF32>> removePerspective =
				new RemovePerspectiveDistortion<Planar<GrayF32>>(1600, 800, ImageType.pl(image.getRaster().getNumBands(), GrayF32.class));

		
		// Specify the corners in the input image of the region.
		// Order matters! top-left, top-right, bottom-right, bottom-left
		//System.out.println("check "+points1.get(0));
		//System.out.println("check "+points1.get(1));
		if( !removePerspective.apply(planar,
				points1.get(0), points1.get(1),
				points1.get(2), points1.get(3)) ){
			throw new RuntimeException("Failed!?!?");
		}
		
		
//		gui2.addImage(image.getSubimage((int)points.get(0).x, (int)points.get(0).y, 300, 300), "Cropped");

		Planar<GrayF32> output = removePerspective.getOutput();
		BufferedImage op = new BufferedImage(1600, 800, image.getType());
		flat = ConvertBufferedImage.convertTo_F32(output,op,true);
		
		gui2.addImage(flat, "After Transform");
		
		nachImageResult = label(flat);
		
		}else if(points.size()==4){
			Planar<GrayF32> planar = ConvertBufferedImage.convertFromPlanar(image, null, true, GrayF32.class);

			//System.out.println(image.getRaster().getNumBands());
//			 gui2.addImage(ConvertBufferedImage.convertTo_F32(planar,null,true), "test");
			RemovePerspectiveDistortion<Planar<GrayF32>> removePerspective =
					new RemovePerspectiveDistortion<Planar<GrayF32>>(1600, 800, ImageType.pl(image.getRaster().getNumBands(), GrayF32.class));

			
			// Specify the corners in the input image of the region.
			// Order matters! top-left, top-right, bottom-right, bottom-left
			if( !removePerspective.apply(planar,
					points.get(0), points.get(1),
					points.get(2), points.get(3)) ){
				throw new RuntimeException("Failed!?!?");
			}
			
//			gui2.addImage(image.getSubimage((int)points.get(0).x, (int)points.get(0).y, 300, 300), "Cropped");

			Planar<GrayF32> output = removePerspective.getOutput();
			//System.out.println(output.getNumBands());
			
			BufferedImage op = new BufferedImage(1600, 800, image.getType());
			BufferedImage flat = ConvertBufferedImage.convertTo_F32(output,op,true);
			
			gui2.addImage(flat, "After Transform");
			
			nachImageResult = label(flat);
		}
		else{
			//System.out.println("Rectangle not detected");
			rectangleDetected = false;
		}
		nachImageResult.setRectangleDetected(rectangleDetected);
		return nachImageResult;
	}
}
