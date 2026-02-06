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
    @SuppressWarnings("null")
    public org.springframework.security.core.userdetails.UserDetails loadUserByUsername(String userId) {
        var u = users.findById(java.util.UUID.fromString(userId))
                .orElseThrow(() -> new org.springframework.security.core.userdetails.UsernameNotFoundException(
                        "User not found"));
        return User.withUsername(u.id.toString()).password(u.passwordHash).authorities("USER").build();
    }
}
