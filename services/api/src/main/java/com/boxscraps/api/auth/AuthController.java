package com.boxscraps.api.auth;

import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService auth;

    AuthController(AuthService auth) {
        this.auth = auth;
    }

    @PostMapping("/register")
    ResponseEntity<AuthResponse> register(@RequestBody RegisterRequest req, HttpServletResponse res) {
        var out = auth.register(req);
        setRefreshCookie(res, out.refreshToken());
        return ResponseEntity.ok(out);
    }

    @PostMapping("/login")
    ResponseEntity<AuthResponse> login(@RequestBody LoginRequest req, HttpServletResponse res) {
        var out = auth.login(req);
        setRefreshCookie(res, out.refreshToken());
        return ResponseEntity.ok(out);
    }

    @PostMapping("/refresh")
    ResponseEntity<AuthResponse> refresh(
            @RequestBody(required = false) RefreshRequest req,
            @CookieValue(name = "rt", required = false) String cookieRt,
            HttpServletResponse res) {
        var token = (req != null && req.refreshToken() != null) ? req.refreshToken() : cookieRt;
        if (token == null)
            return ResponseEntity.badRequest().build();
        var out = auth.refresh(token);
        setRefreshCookie(res, out.refreshToken());
        return ResponseEntity.ok(out);
    }

    @PostMapping("/logout")
    ResponseEntity<Void> logout(
            @RequestBody(required = false) RefreshRequest req,
            @CookieValue(name = "rt", required = false) String cookieRt,
            HttpServletResponse res) {
        var token = (req != null && req.refreshToken() != null) ? req.refreshToken() : cookieRt;
        if (token != null)
            auth.logout(token);
        clearRefreshCookie(res);
        return ResponseEntity.ok().build();
    }

    private void setRefreshCookie(HttpServletResponse res, String refresh) {
        var c = new Cookie("rt", refresh);
        c.setHttpOnly(true);
        c.setSecure(true);
        c.setPath("/api/auth");
        c.setMaxAge(60 * 60 * 24 * 30);
        res.addCookie(c);
    }

    private void clearRefreshCookie(HttpServletResponse res) {
        var c = new Cookie("rt", "");
        c.setHttpOnly(true);
        c.setSecure(true);
        c.setPath("/api/auth");
        c.setMaxAge(0);
        res.addCookie(c);
    }
}
