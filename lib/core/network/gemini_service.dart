import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

import '../../data/models/inference_result.dart';
import '../utils/logger.dart';

final _log = AppLogger('GeminiService');

class GeminiService {
  static const String _baseUrl =
      'https://generativelanguage.googleapis.com/v1';

  static const String _model = 'gemini-1.5-flash-latest';

  final Dio _dio;

  GeminiService({Dio? dio}) : _dio = dio ?? Dio();

  String get _apiKey => dotenv.env['GEMINI_API_KEY'] ?? '';

  static const String _systemPrompt = '''
You are a food nutrition analysis engine. You receive a photo of a food item.
You MUST respond with ONLY a valid JSON object — no markdown, no explanation,
no conversation, no code fences.

The JSON MUST match this exact schema:
{
  "food_item_identified": "<string: name of the food>",
  "confidence_score": <float: 0.0 to 1.0>,
  "calories_kcal": <integer: estimated calories>,
  "proteins_grams": <float: estimated protein in grams>,
  "carbohydrates_grams": <float: estimated carbohydrates in grams>,
  "fats_grams": <float: estimated fats in grams>,
  "serving_size_estimate": "<string: estimated serving size>"
}
''';

  Future<InferenceResult> analyzeFood(
    Uint8List imageBytes, {
    String mimeType = 'image/jpeg',
  }) async {
    _log.info('Sending food image (${imageBytes.length} bytes) to Gemini');

    final base64Image = base64Encode(imageBytes);

    final requestBody = {
      'contents': [
        {
          'parts': [
            {'text': _systemPrompt},
            {
              'inlineData': {'mimeType': mimeType, 'data': base64Image},
            },
          ],
        },
      ],
      'generationConfig': {
        'temperature': 0.1,
        'topP': 0.8,
        'maxOutputTokens': 512,
        'responseMimeType': 'application/json',
      },
    };

    try {
      final response = await _dio.post(
        '$_baseUrl/models/$_model:generateContent',
        queryParameters: {'key': _apiKey},
        data: requestBody,
        options: Options(
          headers: {'Content-Type': 'application/json'},
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      final result = _parseResponse(response.data);

      if (!result.isValid) {
        _log.error('Inference result failed guardrail validation: $result');
        throw InferenceGuardrailException(
          'Inference response contains invalid values: $result',
        );
      }

      _log.info(
        'Inference success: ${result.foodItemIdentified} (${result.confidenceScore} confidence)',
      );
      return result;
    } on DioException catch (e) {
      final dynamic responseData = e.response?.data;
      final String googleError = responseData != null
          ? responseData.toString()
          : (e.message ?? 'Unknown Network Error');

      _log.error('Gemini API request failed. Server response: $googleError');

      throw InferenceNetworkException('Google API Error: $googleError');
    }
  }

  InferenceResult _parseResponse(dynamic responseData) {
    try {
      final candidates = responseData['candidates'] as List<dynamic>;
      if (candidates.isEmpty) {
        throw InferenceParseException('No candidates in Gemini response');
      }

      final content = candidates[0]['content'];
      final parts = content['parts'] as List<dynamic>;
      if (parts.isEmpty) {
        throw InferenceParseException('No parts in Gemini response');
      }

      final rawText = parts[0]['text'] as String;
      final cleanedJson = _stripMarkdownFences(rawText.trim());
      final jsonMap = jsonDecode(cleanedJson) as Map<String, dynamic>;

      return InferenceResult.fromJson(jsonMap);
    } catch (e) {
      if (e is InferenceException) rethrow;
      _log.error('Failed to parse Gemini response', e);
      throw InferenceParseException('Could not parse Gemini response: $e');
    }
  }

  String _stripMarkdownFences(String text) {
    if (text.startsWith('```')) {
      final lines = text.split('\n');
      final filtered = lines
          .where((line) => !line.trim().startsWith('```'))
          .toList();
      return filtered.join('\n').trim();
    }
    return text;
  }
}

sealed class InferenceException implements Exception {
  final String message;
  const InferenceException(this.message);
  @override
  String toString() => '$runtimeType: $message';
}

class InferenceGuardrailException extends InferenceException {
  const InferenceGuardrailException(super.message);
}

class InferenceParseException extends InferenceException {
  const InferenceParseException(super.message);
}

class InferenceNetworkException extends InferenceException {
  const InferenceNetworkException(super.message);
}
