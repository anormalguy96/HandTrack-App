package com.boxscraps.api.web;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

@RestController
@RequestMapping("/api/ads")
class AdsProxyController {

  private final RestClient http;
  private final String adsBase;

  AdsProxyController(@Value("${app.adsServiceBaseUrl}") String adsServiceBaseUrl) {
    this.http = RestClient.create();
    this.adsBase = adsServiceBaseUrl;
  }

  @GetMapping("/config")
  ResponseEntity<String> config() {
    String body = http.get()
        .uri(adsBase + "/ads/config")
        .retrieve()
        .body(String.class);
    return ResponseEntity.ok(body);
  }
}
