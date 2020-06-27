package com.kaleidofin.aadhaar.service;

import java.io.IOException;

public interface AadhaarService {
	
	public String RedactImage(String path) throws IOException;
	
	public String[] GetAadhaarNumber(String desc);
	
}
