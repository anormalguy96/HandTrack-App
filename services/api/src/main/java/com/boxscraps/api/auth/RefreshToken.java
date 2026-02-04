package com.boxscraps.api.auth;

import jakarta.persistence.*;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "refresh_token")
class RefreshToken {

    @Id
    @GeneratedValue
    UUID id;

    @Column(nullable = false)
    UUID userId;

    // store HASHED token only
    @Column(nullable = false, unique = true, length = 64)
    String tokenSha256;

    @Column(nullable = false)
    Instant expiresAt;

    @Column(nullable = false)
    boolean revoked;

    protected RefreshToken() {}

    RefreshToken(UUID userId, String tokenSha256, Instant expiresAt) {
        this.userId = userId;
        this.tokenSha256 = tokenSha256;
        this.expiresAt = expiresAt;
        this.revoked = false;
    }
}
