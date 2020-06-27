package com.kaleidofin.cloudvision;

import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.api.client.googleapis.auth.oauth2.GoogleCredential;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.vision.v1.Vision;
import com.google.api.services.vision.v1.VisionRequestInitializer;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesRequest;
import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.common.collect.ImmutableList;
import com.kaleidofin.ImageText;

public class CloudVisionService {
	
	private final Logger log = LoggerFactory.getLogger(CloudVisionService.class);
	
	public static Vision getVisionService() throws IOException, GeneralSecurityException {
	    GoogleCredential credential =null;
	    JsonFactory jsonFactory = JacksonFactory.getDefaultInstance();
	    return new Vision.Builder(GoogleNetHttpTransport.newTrustedTransport(), jsonFactory, credential)
	            .setApplicationName("kaleidofin").setVisionRequestInitializer(new VisionRequestInitializer("AIzaSyBs7DixzZTGS6c4b3CdPAv_JTAmd0KU4Io"))
	            .build();
	}
	
	
	public List<EntityAnnotation> detectText(byte[] data) {
		Vision vision = null;
		try {
			vision = getVisionService(); 
		}	
		catch (Exception e) {
			log.error(e.toString());
		}
		  
		ImmutableList.Builder<com.google.api.services.vision.v1.model.AnnotateImageRequest> requests = ImmutableList.builder();		
		try {
			requests.add(
					new com.google.api.services.vision.v1.model.AnnotateImageRequest()
					.setImage(new com.google.api.services.vision.v1.model.Image().encodeContent(data))
					.setFeatures(ImmutableList.of(
							new com.google.api.services.vision.v1.model.Feature()
							.setType("TEXT_DETECTION")
							))
					);
	
	
			Vision.Images.Annotate annotate = vision.images().annotate(new BatchAnnotateImagesRequest().setRequests(requests.build()));
			// Due to a bug: requests to Vision API containing large images fail when GZipped.
			annotate.setDisableGZipContent(true);
			com.google.api.services.vision.v1.model.BatchAnnotateImagesResponse batchResponse = annotate.execute();
			//assert batchResponse.getResponses().size() == paths.size();
	
			com.google.api.services.vision.v1.model.AnnotateImageResponse response = batchResponse.getResponses().get(0);
	    
			return response.getTextAnnotations();
			
		} catch (IOException ex) {
			ex.printStackTrace();
			log.error(ex.toString());
			return null;
		}
	}
}

