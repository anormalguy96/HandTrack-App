package com.boxscraps.api.auth;

import jakarta.persistence.*;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "refresh_token")
public class RefreshToken {

    @Id
    @GeneratedValue
    public UUID id;

    @Column(nullable = false)
    public UUID userId;

    // store HASHED token only
    @Column(nullable = false, unique = true, length = 64)
    public String tokenSha256;

    @Column(nullable = false)
    public Instant expiresAt;

    @Column(nullable = false)
    public boolean revoked;

    protected RefreshToken() {
    }

    RefreshToken(UUID userId, String tokenSha256, Instant expiresAt) {
        this.userId = userId;
        this.tokenSha256 = tokenSha256;
        this.expiresAt = expiresAt;
        this.revoked = false;
    }
}
