import 'dart:convert';
import 'package:http/http.dart' as http;
import 'env.dart';

class AdsClient {
  Future<Map<String, dynamic>> getAdsConfig() async {
    final uri = Uri.parse('${Env.adsServiceBaseUrl}/ads/config');
    final res = await http.get(uri);
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception('Ads config failed: ${res.statusCode}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }
}
