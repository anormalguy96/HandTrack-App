package com.boxscraps.api.web;

import java.time.Instant;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
class HealthController {

  @GetMapping("/health")
  Object health() {
    return new HealthResponse(true, Instant.now().toString());
  }

  record HealthResponse(boolean ok, String now) {}
}
