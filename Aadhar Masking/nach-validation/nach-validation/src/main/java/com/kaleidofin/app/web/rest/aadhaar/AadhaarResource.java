package com.kaleidofin.app.web.rest.aadhaar;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.sql.SQLException;
import java.text.ParseException;

import javax.imageio.ImageIO;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.support.StandardMultipartHttpServletRequest;

import com.kaleidofin.ModelBuilderGUI;
import com.kaleidofin.NachImageResult;
import com.kaleidofin.cloudvision.CloudVisionService;

@RestController
@RequestMapping("/aadhaar")
public class AadhaarResource {
	
	private final Logger log = LoggerFactory.getLogger(AadhaarResource.class);

	@PostMapping("/redact/")
	public String getImage(StandardMultipartHttpServletRequest request) throws ParseException, IOException, SQLException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{
    	long start = System.currentTimeMillis();
		MultipartFile multipartImage = request.getFile("AadhaarImage");
		InputStream in = new ByteArrayInputStream(multipartImage.getBytes());
		BufferedImage bImageFromConvert = ImageIO.read(in);
		NachImageResult nachImageResult = ModelBuilderGUI.preprocessing(bImageFromConvert);
		long end = System.currentTimeMillis();
		nachImageResult.setTimeTaken(end-start);
		return nachImageResult.toString();
	}
	
	@GetMapping("/redactTest/")
	public void getImageTest() throws ParseException, IOException, SQLException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{
    	long start = System.currentTimeMillis();
    	log.info("Redact AAdhaar Image");
    	File img = new File("C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\Img\\1e053f25-e522-4fd1-a2db-5a4b5c934231.jpg");
    	BufferedImage bufferedImage = ImageIO.read(img);
    	ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(bufferedImage, "jpg", baos);
		byte[] srcData = baos.toByteArray(); 
		CloudVisionService cloudVision = new CloudVisionService();
		cloudVision.detectText(srcData);
		long end = System.currentTimeMillis();
		log.info("Time Taken: {}", (end-start));
	}
	
	@GetMapping("/hello/")
	public String Hello() {
		log.info("Hello");
		return "Hello";
	}
}
