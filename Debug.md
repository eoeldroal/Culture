# DPO Trainer와 LlavaOnevision 모델 통합 디버깅 기록

## 문제 발생 일자
2024년 기준 작성

## 문제 상황 요약

VARCO-VISION-2.0-14B 모델을 사용하여 MPO (Multi-Preference Optimization) 훈련을 구현하던 중, DPOTrainer 초기화 단계에서 AttributeError가 발생했습니다. 이 문제는 데이터셋을 토크나이징하는 과정에서 발생했으며, 구체적으로는 이미지 프로세서가 패딩 작업을 수행하려 할 때 리스트 객체에서 shape 속성을 찾을 수 없다는 오류였습니다.

## 상세 오류 내용

### 오류 메시지
```
AttributeError: 'list' object has no attribute 'shape'
```

### 오류 발생 위치
오류는 `LlavaOnevisionImageProcessorFast` 클래스의 `_pad_for_batching` 메서드에서 발생했습니다. 구체적으로 다음 코드 라인에서 문제가 발생했습니다:

```python
torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
```

프로세서는 image 변수가 PyTorch 텐서일 것으로 예상했지만, 실제로는 Python 리스트가 전달되어 shape 속성에 접근할 수 없었던 것이 직접적인 원인이었습니다.

## 문제의 근본 원인 분석

이 문제는 여러 층위의 원인이 복합적으로 작용한 결과였습니다. 먼저 가장 표면적인 원인은 Fast 프로세서와 일반 프로세서 간의 구현 차이였습니다. Fast 프로세서는 성능 최적화를 위해 Rust로 구현되어 있으며, 타입 체크를 더 엄격하게 수행합니다. 반면 일반 프로세서는 Python으로 구현되어 있어 더 유연한 타입 처리가 가능합니다.

더 깊은 원인을 살펴보면, LlavaOnevision 모델의 복잡한 이미지 처리 파이프라인과 DPO Trainer의 기대 사항 간에 불일치가 있었습니다. LlavaOnevision은 이미지를 여러 패치로 나누어 처리하는 정교한 시스템을 가지고 있는데, 이 과정에서 데이터의 형태가 여러 번 변환됩니다. DPO Trainer는 원래 텍스트 전용 모델을 위해 설계되었기 때문에, 이러한 복잡한 비전 처리 파이프라인과의 통합에서 예상치 못한 문제가 발생할 수 있습니다.

또한 이 조합(DPO Trainer + LlavaOnevision + Fast Processor)은 상대적으로 새로운 것이어서 충분한 테스트가 이루어지지 않았을 가능성이 높습니다. 특히 Fast 프로세서는 주로 텍스트 모델을 위해 먼저 최적화되었고, 복잡한 비전-언어 모델에 대한 지원은 아직 완벽하지 않을 수 있습니다.

## 해결 방법

문제는 의외로 간단한 방법으로 해결되었습니다. AutoProcessor를 초기화할 때 `use_fast` 파라미터를 명시적으로 False로 설정하거나, 아예 생략하여 기본값(False)을 사용하도록 했습니다:

```python
# 문제가 있던 코드
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

# 해결된 코드
processor = AutoProcessor.from_pretrained(MODEL_ID)  # use_fast=False가 기본값
```

이 변경으로 Fast 프로세서 대신 일반 프로세서가 로드되었고, 일반 프로세서의 더 유연한 타입 처리 덕분에 문제가 해결되었습니다. 일반 프로세서는 입력 데이터가 리스트 형태로 들어와도 적절히 텐서로 변환하는 로직을 포함하고 있어, 다양한 데이터 형태를 안정적으로 처리할 수 있습니다.

## 향후 예방 방법

### 1. 새로운 모델 조합 사용 시 체크리스트

새로운 비전-언어 모델을 사용할 때는 다음 순서로 접근하는 것이 안전합니다. 먼저 일반 프로세서(use_fast=False)로 시작하여 기본 기능이 작동하는지 확인합니다. 모든 것이 안정적으로 작동한다면, 그 다음에 Fast 프로세서를 시도해보고 성능 향상이 있는지 측정합니다. 만약 Fast 프로세서에서 오류가 발생한다면, 일반 프로세서를 계속 사용하는 것이 현명합니다.

### 2. 프로세서 호환성 테스트 코드

프로젝트 초기에 다음과 같은 테스트 코드를 실행하여 프로세서가 올바르게 작동하는지 확인하는 것이 좋습니다:

```python
def test_processor_compatibility(model_id, sample_dataset):
    """프로세서 호환성을 테스트하는 함수"""
    from PIL import Image
    
    # 테스트용 더미 이미지 생성
    dummy_image = Image.new('RGB', (224, 224))
    dummy_text = "This is a test prompt"
    
    # 일반 프로세서 테스트
    print("일반 프로세서 테스트 중...")
    try:
        processor_normal = AutoProcessor.from_pretrained(model_id, use_fast=False)
        result = processor_normal(images=dummy_image, text=dummy_text)
        print(f"✓ 일반 프로세서 정상 작동 - 클래스: {type(processor_normal).__name__}")
    except Exception as e:
        print(f"✗ 일반 프로세서 오류: {e}")
    
    # Fast 프로세서 테스트
    print("\nFast 프로세서 테스트 중...")
    try:
        processor_fast = AutoProcessor.from_pretrained(model_id, use_fast=True)
        result = processor_fast(images=dummy_image, text=dummy_text)
        print(f"✓ Fast 프로세서 정상 작동 - 클래스: {type(processor_fast).__name__}")
    except Exception as e:
        print(f"✗ Fast 프로세서 오류: {e}")
        print("  → 일반 프로세서 사용을 권장합니다")
    
    # 실제 데이터셋 샘플로 테스트 (선택사항)
    if sample_dataset is not None:
        print("\n실제 데이터 샘플 테스트 중...")
        sample = sample_dataset[0]
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            result = processor(images=sample['images'], text=sample['prompt'])
            print(f"✓ 실제 데이터 처리 성공")
        except Exception as e:
            print(f"✗ 실제 데이터 처리 실패: {e}")
```

### 3. 디버깅 전략

비전-언어 모델 관련 오류를 만났을 때는 체계적인 접근이 필요합니다. 가장 중요한 것은, 데이터의 형태를 점검하는 것입니다. 이미지가 예상한 형태(PIL Image, 텐서, 리스트 등)로 되어 있는지 확인합니다. 