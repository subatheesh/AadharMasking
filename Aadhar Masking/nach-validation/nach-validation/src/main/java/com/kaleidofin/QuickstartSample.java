package com.kaleidofin;

import java.io.IOException;
import java.security.GeneralSecurityException;

import com.google.api.client.googleapis.auth.oauth2.GoogleCredential;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.vision.v1.Vision;
import com.google.api.services.vision.v1.VisionRequestInitializer;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesRequest;
import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.common.collect.ImmutableList;

public class QuickstartSample {
	
	  public static Vision getVisionService() throws IOException, GeneralSecurityException {
		    GoogleCredential credential =null;
		    JsonFactory jsonFactory = JacksonFactory.getDefaultInstance();
		    return new Vision.Builder(GoogleNetHttpTransport.newTrustedTransport(), jsonFactory, credential)
		            .setApplicationName("kaleidofin").setVisionRequestInitializer(new VisionRequestInitializer("AIzaSyBs7DixzZTGS6c4b3CdPAv_JTAmd0KU4Io"))
		            .build();
	  }
	  
	  
	  public String detectText(byte[] data) {
		  Vision vision = null;
		  
		  try {
			   vision = getVisionService();
			  
		  }
		  catch (Exception e) {
			  e.printStackTrace();
		}
		  
		  
		  ImmutableList.Builder<com.google.api.services.vision.v1.model.AnnotateImageRequest> requests = ImmutableList.builder();		
		    try {
		    	
//		        byte[] data;
		        requests.add(
		            new com.google.api.services.vision.v1.model.AnnotateImageRequest()
		                .setImage(new com.google.api.services.vision.v1.model.Image().encodeContent(data))
		                .setFeatures(ImmutableList.of(
		                    new com.google.api.services.vision.v1.model.Feature()
		                        .setType("TEXT_DETECTION")
		                        )));


		      Vision.Images.Annotate annotate =
		          vision.images()
		              .annotate(new BatchAnnotateImagesRequest().setRequests(requests.build()));
		      // Due to a bug: requests to Vision API containing large images fail when GZipped.
		      annotate.setDisableGZipContent(true);
		      com.google.api.services.vision.v1.model.BatchAnnotateImagesResponse batchResponse = annotate.execute();
		      //assert batchResponse.getResponses().size() == paths.size();

		      ImmutableList.Builder<ImageText> output = ImmutableList.builder();
		        com.google.api.services.vision.v1.model.AnnotateImageResponse response = batchResponse.getResponses().get(0);
		        
		        String text = null;
		        System.out.println(response.getError()+" error"+" "+response.getTextAnnotations());
		        
		        if(response.getTextAnnotations()!=null) {
		            for (EntityAnnotation annotation : response.getTextAnnotations()) {
		            	text = annotation.getDescription();
		            }
		        	
		        }
		        return text;
		    } catch (IOException ex) {
		    	ex.printStackTrace();
		      // Got an exception, which means the whole batch had an error.
		      ImmutableList.Builder<ImageText> output = ImmutableList.builder();
		      return null;
		    }
		  }

	}
