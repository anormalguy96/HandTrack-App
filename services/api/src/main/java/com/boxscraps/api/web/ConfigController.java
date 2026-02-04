package com.boxscraps.api.web;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
class ConfigController {

  @Value("${app.adsServiceBaseUrl}")
  String adsServiceBaseUrl;

  @Value("${app.privacy.uploadFrames:false}")
  boolean uploadFrames;

  @GetMapping("/config")
  Object config() {
    return new ConfigResponse("0.1.0", adsServiceBaseUrl, uploadFrames);
  }

  record ConfigResponse(String version, String adsServiceBaseUrl, boolean uploadFrames) {}
}
