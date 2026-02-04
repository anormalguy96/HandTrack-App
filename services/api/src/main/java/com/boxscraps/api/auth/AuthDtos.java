package com.boxscraps.api.auth;

record RegisterRequest(String email, String password, String displayName) {}
record LoginRequest(String email, String password) {}
record RefreshRequest(String refreshToken) {}
record AuthResponse(String accessToken, String refreshToken) {}
