package com.boxscraps.api.users;

import jakarta.persistence.*;

import java.util.UUID;

@Entity
@Table(name = "app_user")
class AppUser {

    @Id
    @GeneratedValue
    UUID id;

    @Column(nullable = false, unique = true)
    String email;

    @Column(nullable = false)
    String passwordHash;

    @Column(nullable = false)
    String displayName;

    protected AppUser() {}

    AppUser(String email, String passwordHash, String displayName) {
        this.email = email;
        this.passwordHash = passwordHash;
        this.displayName = displayName;
    }
}