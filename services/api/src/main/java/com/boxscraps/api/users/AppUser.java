package com.boxscraps.api.users;

import jakarta.persistence.*;

import java.util.UUID;

@Entity
@Table(name = "app_user")
public class AppUser {

    @Id
    @GeneratedValue
    public UUID id;

    @Column(nullable = false, unique = true)
    public String email;

    @Column(nullable = false)
    public String passwordHash;

    @Column(nullable = false)
    public String displayName;

    public AppUser() {
    }

    public AppUser(String email, String passwordHash, String displayName) {
        this.email = email;
        this.passwordHash = passwordHash;
        this.displayName = displayName;
    }
}