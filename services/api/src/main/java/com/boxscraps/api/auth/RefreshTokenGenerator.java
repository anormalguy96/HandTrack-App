package com.boxscraps.api.auth;

import java.security.SecureRandom;
import java.util.Base64;

final class RefreshTokenGenerator {
    private static final SecureRandom RNG = new SecureRandom();

    static String newToken() {
        var b = new byte[48];
        RNG.nextBytes(b);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(b);
    }

    private RefreshTokenGenerator() {}
}
