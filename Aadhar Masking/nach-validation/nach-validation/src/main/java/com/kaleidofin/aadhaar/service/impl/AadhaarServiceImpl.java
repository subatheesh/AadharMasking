package com.kaleidofin.aadhaar.service.impl;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.api.services.vision.v1.model.Vertex;
import com.kaleidofin.aadhaar.service.AadhaarService;
import com.kaleidofin.cloudvision.CloudVisionService;

import boofcv.gui.feature.VisualizeShapes;
import boofcv.io.image.UtilImageIO;
import georegression.struct.shapes.Polygon2D_F64;

public class AadhaarServiceImpl implements AadhaarService {

	@Override
	public String RedactImage(String path) throws IOException {
//		File img = new File("C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\Img\\1e053f25-e522-4fd1-a2db-5a4b5c934231.jpg");
		File img = new File(path);
    	BufferedImage bufferedImage = ImageIO.read(img);
    	ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(bufferedImage, "jpg", baos);
		byte[] srcData = baos.toByteArray(); 
		CloudVisionService cloudVision = new CloudVisionService();
		List<EntityAnnotation> DetectedTexts = cloudVision.detectText(srcData);
		
		if(DetectedTexts != null) {
			String[] AadhaarNumber = GetAadhaarNumber(DetectedTexts.get(0).getDescription());
			
			if(AadhaarNumber != null) {
				int index = 0, aadhaarSetCount = AadhaarNumber.length, k = 0;
				for(EntityAnnotation annotation: DetectedTexts) {
					if(annotation.getDescription().equals(AadhaarNumber[index])) 
						index++;
					else
						index = 0;
					
					if(index == aadhaarSetCount) {
						int totalChars = 8;
						for(int i=0; i<aadhaarSetCount; i++) {
							double ratio = 1.0;
							EntityAnnotation cur = DetectedTexts.get(k+i-aadhaarSetCount-1);
							int prevTotal = totalChars, curLength = cur.getDescription().length();
							if(totalChars < curLength){
								ratio = (double) totalChars / (double) curLength;
								totalChars = 0;
							}else {
								totalChars -= curLength;
							}
							if(prevTotal > 0) {
								double[][] vertices = getVertices(cur.getBoundingPoly().getVertices());
								if(ratio < 1.0) {
									vertices[1][0] = vertices[0][0] + ratio*(vertices[1][0]-vertices[0][0]);
									vertices[1][1] = vertices[0][1] + ratio*(vertices[1][1]-vertices[0][1]);
			                        vertices[2][0] = vertices[3][0] + ratio*(vertices[2][0]-vertices[3][0]);
			                        vertices[2][1] = vertices[3][1] + ratio*(vertices[2][1]-vertices[3][1]);
								}
								Polygon2D_F64 polygon = new Polygon2D_F64(vertices);
								VisualizeShapes.fillPolygon(polygon, 1, bufferedImage.createGraphics());
							}
						}
						index = 0;
					}
				}
				UtilImageIO.saveImage(bufferedImage, path);
			}
		}
		return null;
	}
	
	@Override
	@SuppressWarnings({ "unused", "null" })
	public String[] GetAadhaarNumber(String desc){
		List<String> Aadhaar = new ArrayList<String>();
		String[] lines = desc.split("\n");
		for(String line: lines) {
			if(checkNumbers(line) == 12){
				Aadhaar.add(line);
			}
			
		}
		if(Aadhaar != null) {
			return Aadhaar.get(0).split(" ");
		}
		return null;
	}
	
	public int checkNumbers(String text) {
		int count = 0;
		for(char c : text.toCharArray()) {
			if(c == 10 || c == 32)	
				continue;
			else if(c > 47 && c < 58)
				count++;
			else
				return 0;
		}
		return count;
	}

	
	public double[][] getVertices(List<Vertex> vertices) {
		double[][] vertices2D = new double[4][2]; int i=0;
		for(Vertex vertex: vertices) {
			vertices2D[i][0] = vertex.getX();
			vertices2D[i][1] = vertex.getY();
			i++;
		}
		return vertices2D;
	}
}
