// Starter environment config.
// Later: use --dart-define or a typed config system.
class Env {
  static const apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8080', // Android emulator -> host
  );

  static const adsServiceBaseUrl = String.fromEnvironment(
    'ADS_BASE_URL',
    defaultValue: 'http://10.0.2.2:8090',
  );
}
