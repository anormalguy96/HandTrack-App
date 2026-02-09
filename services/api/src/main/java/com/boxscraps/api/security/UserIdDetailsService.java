package com.boxscraps.api.security;

import com.boxscraps.api.users.AppUserRepository;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;

@Service
class UserIdDetailsService implements UserDetailsService {

    private final AppUserRepository users;

    UserIdDetailsService(AppUserRepository users) {
        this.users = users;
    }

    @Override
    public org.springframework.security.core.userdetails.UserDetails loadUserByUsername(String userId) {
        if (userId == null) {
            throw new org.springframework.security.core.userdetails.UsernameNotFoundException("User id is null");
        }
        var u = users.findById(java.util.Objects.requireNonNull(java.util.UUID.fromString(userId)))
                .orElseThrow(() -> new org.springframework.security.core.userdetails.UsernameNotFoundException(
                        "User not found"));
        return User.withUsername(u.id.toString()).password(u.passwordHash).authorities("USER").build();
    }
}
