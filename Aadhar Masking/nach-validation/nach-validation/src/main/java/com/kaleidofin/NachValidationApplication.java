package com.kaleidofin;


import java.net.InetAddress;
import java.net.UnknownHostException;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.config.server.EnableConfigServer;

@SpringBootApplication
public class NachValidationApplication {

	public static void main(String[] args) throws UnknownHostException {
		SpringApplication.run(NachValidationApplication.class, args);
		System.out.println("Microservice Started");
	}
}
