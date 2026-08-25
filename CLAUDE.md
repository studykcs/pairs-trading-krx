## 프로젝트 개요
KRX Open API로 KOSPI 일별 종가를 수집하고, 공적분 스크리닝으로 페어를 찾아
z-score 평균회귀 전략을 백테스트하는 리서치 파이프라인.

## 현재 파일
- `collect.py`        : KRX Open API 일별매매정보 수집
- `store.py`          : SQLite (prices, tickers), long→wide 변환
- `cointegration.py`  : 1 target vs N candidates Engle-Granger + FDR
- `screen_basket.py`  : 바스켓 내 C(n,2) 전수 검정
- `backtest.py`       : static / walk-forward beta z-score 백테스트
- `gmm_strategy.py`   : GMM 레짐 필터
- `strategy.py`       : 비용 + 스탑로스 포함 최종 전략

## 이 리포의 핵심 결함 (리팩터링 목표)
1. **페어 선택 편향**: cointegration.py가 전체 표본으로 페어를 고른 뒤
   backtest.py가 같은 표본으로 백테스트한다. 베타는 walk-forward인데
   "어떤 페어를 트레이드할지"라는 결정 자체에 미래 정보가 들어간다.
   → 이게 가장 큰 문제. rolling formation/trading 분할로 해결해야 한다.
2. **다중검정 미보정 성과**: 924개 후보 중 최고 샤프를 보고한다.
   p-value에는 FDR을 적용했지만 성과 지표에는 같은 논리를 적용하지 않았다.
3. **귀무모형 부재**: 공적분 스크리닝이 실제로 무언가를 하는지 검증한 적이 없다.
4. **단일 페어**: 페어 하나는 사례이지 전략이 아니다. 포트폴리오 레벨이 없다.
5. **한국 시장 제약 미반영**: 공매도 가능 종목 제한, 매도 거래세, 대차 비용,
   상하한가, 거래정지가 전혀 모델링되지 않았다.
6. **GMM 라벨 불안정**: 날짜별 독립 분류라 상태 지속성 개념이 없고,
   재추정마다 calm 컴포넌트를 다시 골라 label switching에 취약하다.

## 알려진 버그
- `backtest.py::_summarize` — spread_ret은 로그수익률인데
  `(1+r).cumprod()`로 단순수익률처럼 복리 계산한다. `np.exp(r.cumsum())`이 맞다.
- `backtest.py::run_backtest_static` — `hedge_ratio`가 None을 반환하면
  `beta, _ = ...` 언패킹에서 터진다. walk-forward 쪽만 None을 처리한다.

## 서사 원칙 (중요)
이 리포의 결과물은 "샤프 2.1이 나왔다"가 아니라
"룩어헤드를 단계적으로 제거했더니 성과가 어떻게 무너졌고, 왜 무너졌는가"다.
성과 악화는 실패가 아니라 산출물이다. 기존 -98% 드로다운 결과를 지우지 말 것.

## 코딩 규약
- Python 3.11+, `from __future__ import annotations`
- 실행 스크립트는 argparse + 모듈 docstring에 Usage 섹션
- docstring에 "왜 이 방법을 택했는가"를 반드시 적을 것
- 기존 파일을 삭제하지 말고 확장할 것 (기존 결과를 재현할 수 있어야 함)
- 무거운 계산은 결과를 캐시(parquet/sqlite)하고 재실행 시 재사용

## 하지 말 것
- 성과가 좋아 보이도록 파라미터/기간/유니버스를 조정하지 말 것
- 결과가 부정적이면 부정적으로 보고할 것
- 랜덤 시드를 고정하지 않은 채 결과를 보고하지 말 것
