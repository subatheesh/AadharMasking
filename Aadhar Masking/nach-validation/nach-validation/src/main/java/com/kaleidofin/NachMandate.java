package com.kaleidofin;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.SQLException;
import java.text.ParseException;

import javax.imageio.ImageIO;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.support.StandardMultipartHttpServletRequest;


@RestController
@RequestMapping("/nachMandate")
public class NachMandate {
	
    @PostMapping("/validate/")
	public NachImageResult getImage(StandardMultipartHttpServletRequest request) throws ParseException, IOException, SQLException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{
    	long start = System.currentTimeMillis();
		MultipartFile multipartImage = request.getFile("nachImage");
		InputStream in = new ByteArrayInputStream(multipartImage.getBytes());
		BufferedImage bImageFromConvert = ImageIO.read(in);
		NachImageResult nachImageResult = ModelBuilderGUI.preprocessing(bImageFromConvert);
		long end = System.currentTimeMillis();
		nachImageResult.setTimeTaken(end-start);
		return nachImageResult;
	}
    

	@GetMapping("/hello/")
	public String Hello() {
		return "Hello";
	}
}
