package com.boxscraps.api.auth;

import com.boxscraps.api.security.JwtService;
import com.boxscraps.api.users.AppUser;
import com.boxscraps.api.users.AppUserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.time.Instant;
import java.util.HexFormat;
import java.util.UUID;

@Service
public class AuthService {

    private final AppUserRepository users;
    private final RefreshTokenRepository refreshTokens;
    private final PasswordEncoder encoder;
    private final JwtService jwtService;

    AuthService(AppUserRepository users, RefreshTokenRepository refreshTokens, PasswordEncoder encoder,
            JwtService jwtService) {
        this.users = users;
        this.refreshTokens = refreshTokens;
        this.encoder = encoder;
        this.jwtService = jwtService;
    }

    public AuthResponse register(RegisterRequest req) {
        users.findByEmailIgnoreCase(req.email()).ifPresent(u -> {
            throw new IllegalArgumentException("Email already used");
        });
        var user = users
                .save(new AppUser(req.email().toLowerCase(), encoder.encode(req.password()), req.displayName()));
        return issueTokens(user.id);
    }

    public AuthResponse login(LoginRequest req) {
        var user = users.findByEmailIgnoreCase(req.email())
                .orElseThrow(() -> new IllegalArgumentException("Invalid credentials"));
        if (!encoder.matches(req.password(), user.passwordHash)) {
            throw new IllegalArgumentException("Invalid credentials");
        }
        return issueTokens(user.id);
    }

    public AuthResponse refresh(String rawRefreshToken) {
        var hash = sha256(rawRefreshToken);
        var rt = refreshTokens.findByTokenSha256(hash)
                .orElseThrow(() -> new IllegalArgumentException("Invalid refresh token"));
        if (rt.revoked || rt.expiresAt.isBefore(Instant.now())) {
            throw new IllegalArgumentException("Refresh token expired");
        }
        return new AuthResponse(jwtService.issueAccessToken(rt.userId.toString()), rawRefreshToken);
    }

    public void logout(String rawRefreshToken) {
        var hash = sha256(rawRefreshToken);
        refreshTokens.findByTokenSha256(hash).ifPresent(rt -> {
            rt.revoked = true;
            refreshTokens.save(rt);
        });
    }

    private AuthResponse issueTokens(UUID userId) {
        var access = jwtService.issueAccessToken(userId.toString());
        var refresh = RefreshTokenGenerator.newToken();
        var refreshHash = sha256(refresh);
        refreshTokens.save(new RefreshToken(userId, refreshHash, Instant.now().plusSeconds(60L * 60 * 24 * 30))); // 30d
        return new AuthResponse(access, refresh);
    }

    static String sha256(String s) {
        try {
            var md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(s.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
