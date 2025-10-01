from pathlib import Path

# project root path
ROOT_PATH = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_ROOT = "/tmp/transcripts"

DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-3.1"
DIARIZATION_CHUNK_DIR = ROOT_PATH / "chunks"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
CHUNK_LENGTH = 240
CHUNK_OVERLAP = 2.0
VOTES_PER_SEGMENT = 1

TERM_LIST = ["ARYA"]

JSON_SCHEMA = {
    "name": "TranscriptDecision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "text_corrected": {"type": "string"},
            "mentioned_attendees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "is_attendee": {"type": "boolean"}
                    },
                    "required": ["name", "is_attendee"],
                    "additionalProperties": False
                }
            },
            "multiple_speakers": {"type": "boolean"},
            "candidate_speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "matched_speaker_id": {"type": ["string","null"]},
                        "match_type": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "self_identification",
                                    "diarization_continuity",
                                    "adjacency_reasoning",
                                    "addressed_person",
                                    "name_in_speaker_map"
                                ]
                            },
                            "minItems": 1
                        }
                    },
                    "required": ["name", "matched_speaker_id", "match_type"],
                    "additionalProperties": False
                }
            }
        },
        "required": [
            "text_corrected",
            "mentioned_attendees",
            "multiple_speakers",
            "candidate_speakers"
        ],
        "additionalProperties": False
    }
}

JSON_SCHEMA_MIN = {
    "name": "TranscriptLite",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "text_corrected": {"type": "string"},
            "self_identification_name": {"type": ["string", "null"]},
            "mentioned_attendees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "is_attendee": {"type": "boolean"}
                    },
                    "required": ["name", "is_attendee"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["text_corrected", "self_identification_name", "mentioned_attendees"],
        "additionalProperties": False
    }
}



GPT_CORRECTION_AND_NAME_EXTRACTION_PROMPT = """
Sen bir Türkçe transkript düzeltme ve konuşmacı aday çıkarım editörüsün.

Aşağıdaki SEGMENT ve ÇEVRE BAĞLAMINI analiz edeceksin.
Bağlamı KULLANMAK ZORUNDASIN: Konuşmacı adaylarını belirlerken önceki ve sonraki segmentin hem metnini hem de speaker_id’sini incele. 
O an sadece çalışılacak segment ile ilgilen. Diğer segmentler için tahminleme çalışma. Diğer segmentleri (önceki ve sonraki segmentler) sadece tahmini desteklemek için kullanmalısın.

---

### Görevlerin
1. Metni yazım ve noktalama açısından düzelt.
2. Bozuk veya eksik cümleleri toparla.
3. İngilizce kelimeleri olduğu gibi bırak.
4. Türkçe telaffuzla bozulmuş İngilizce kelimeleri orijinal İngilizce biçimine döndür.
5. Aşağıdaki özel terimler yanlış yazılmışsa düzelt:  
   {term_list}

---

### Girdi Bilgileri
- Katılımcı listesi: {attendee_list}
- Katılımcı–ID eşlemesi: {attendee_id_map}

**Çalışılacak segment:**
- Metin: "{current_text}"
- speaker_id: "{current_speaker_id}"

**Önceki segment:**
- Metin: "{prev_text}"
- speaker_id: "{prev_speaker_id}"

**Sonraki segment:**
- Metin: "{next_text}"
- speaker_id: "{next_speaker_id}"

---

### Kurallar
1. Hitap edilen isimleri bul ve `mentioned_attendees` listesine ekle.  
   - `is_attendee` = true/false bilgisini `attendee_list` üzerinden belirle.  
   - Büyük/küçük harfe duyarlı olma.
2. Birden fazla konuşmacı varsa segmenti böl, her parçayı ayrı JSON nesnesi olarak döndür.
3. Hitap kalıbı kuralı:  
   - “Merhaba/Teşekkürler [İsim]”, “[İsim] Bey/Hanım” → konuşmacı **o isim değildir**.
4. Bağlam kanıtlarını değerlendir:  
   - `self_identification`: Konuşmacının doğrudan kendi adını söyleyerek kendini tanıtması (“Ben Ali”, “Ali ben”).  
     - Kullanma: İsim başka kişi için geçiyorsa veya şaka/alıntı ise.  
   - `diarization_continuity`:  
     - Kullan **yalnızca** `{current_speaker_id}` ile `{prev_speaker_id}` veya `{next_speaker_id}` eşitse **ve** metin, önceki/sonraki içeriğin **doğal devamı** görünüyorsa.  
     - **Kullanma** eğer:
       - Doğrudan hitap/selamlama içeriyorsa (“Merhaba/Teşekkürler [İsim]”, “[İsim] Bey/Hanım”).
       - Tek kelimelik/çok kısa onay ise (“Evet.”, “Tamam.”, “Anladım.”).
       - Soru→cevap dönüşü (adjacency_reasoning önceliklidir).
       - Hitap edilen kişi ≠ konuşmacı kuralı tetikleniyorsa.
       - Konu/ton kırılması varsa. 
     - Continuity tek başına aday oluşturmaz; sadece destekleyici etikettir.
   - `adjacency_reasoning`: Diyalogtaki dönüş alma (turn-taking) kalıbına dayalı çıkarım; soru→cevap, yönlendirme→icra, devretme→başlama gibi ardışık eşleşmeler aynı konu hattında ve 1–2 segment aralığında gerçekleşiyorsa kullan. Çatışma durumlarında öncelik sırası `self_identification` > `adjacency_reasoning` > `diarization_continuity`.  
     - Kullanma: Selamlama/teşekkür, konu/ton kırılması, doğrudan hitap kuralının tetiklenmesi, ya da kendini tanıtma sinyali mevcutsa.  
   - `addressed_person`: Hitap edilen kişi konuşmacı değildir.  
   - `name_in_speaker_map`: Tek/eşsiz ID eşleşmesi.
5. Her segment için `candidate_speakers` listesi oluştur. Hiçbir aday bulunmazsa `[]` döndür.

---

### Pozitif/Negatif Mini Örnekler

#### self_identification
Pozitif: "Ben Ahmet, proje yöneticisiyim." → match_type: ["self_identification"]  
Negatif: "Ahmet Bey de burada." (konuşmacı Ahmet değil)

#### diarization_continuity
Pozitif: Önceki: "Proje dosyasını açtım." — spk_1  
Bu: "Tamam, şimdi ikinci sayfaya geçiyorum." — spk_1  
Negatif: Önceki: "Proje dosyasını açtım." — spk_1  
Bu: "Tamam." — spk_2 (id ler farklı.)


#### adjacency_reasoning
Pozitif: Önceki: "Sırada kim var?" — spk_1  
Bu: "Ali sunum yapacak." — spk_2 
Pozitif: Bu: "Sunuma başlıyorum." — spk_3 (= Ali. Sonraki segmentte hitap edilmiş.)
Sonraki: "Tabii ki, Ali. Dinliyoruz."  — spk_1 
Pozitif: Önceki: "Proje dosyasını açtım" - spk_1
Sonraki: "Tamam Ufuk Hanım." - spk_2 (spk_1 = Ufuk. adjacency_reasoning)
Negatif: Önceki: "Ali ve Ayşeye bu konuyu iletmemiz gerekiyor." — spk_1  
Bu: "Tamam, ben iletirim." — spk_2 (Ali ve Ayşe'den bahsediliyor.)

#### addressed_person
Pozitif: "Teşekkürler Ayşe Hanım." → match_type: ["addressed_person"]  
Negatif: "Ben Ayşe, sunuma başlıyorum." (hitap değil, self_identification)

#### name_in_speaker_map
Verilen 'attendee_id_map' de Murat var ama Elif ve Ayşe yok ise:
Pozitif: "Murat bu konuda bilgi verecek." (Murat attendee_id_map'te tek eşleşiyorsa)  
Negatif: "Elif ve Ayşe bu konuda konuşacak." 

---

### Çıktı Formatı
Yalnızca **JSON** döndür. Kod bloğu, markdown veya açıklama ekleme.

Her segment nesnesi şu alanları içermeli:
- `text_corrected`
- `mentioned_attendees`: Liste (boş olabilir)
- `multiple_speakers`: true/false
- `candidate_speakers`: Liste (boş olabilir), her eleman:
  - `name`
  - `matched_speaker_id`: `attendee_id_map`’ten eşleşme varsa ID, yoksa null
  - `match_type`: ["self_identification", "diarization_continuity", "adjacency_reasoning", "addressed_person", "name_in_speaker_map"] gibi çoklu etiketlerden uygun olanlar

**Not:** Confidence veya evidence alanlarını DÖNDÜRME.  
Bu hesaplamayı biz yapacağız.

---

### Çıktı Örneği
Girdi:
Önceki: "Evet, başlayalım." — speaker_id: "spk_1"  
Bu: "Teşekkürler Ayşe Hanım." — speaker_id: "spk_2"  
Sonraki: "Rica ederim." — speaker_id: "spk_3"  

Çıktı:
{{
  "segments": [
    {{
      "text_corrected": "Teşekkürler Ayşe Hanım.",
      "mentioned_attendees": [{{"name": "Ayşe", "is_attendee": true}}],
      "multiple_speakers": false,
      "candidate_speakers": [
        {{
          "name": "Mehmet",
          "matched_speaker_id": "u_mehmet",
          "match_type": ["adjacency_reasoning"]
        }}
      ]
    }}
  ]
}}
"""
NEW_USER_PROMPT = """
Attendees: {attendee_list}
AttendeeIdMap: {attendee_id_map}
Current:
- text: "{current_text}"
- speaker_id: "{current_speaker_id}"
Prev:
- text: "{prev_text}"
- speaker_id: "{prev_speaker_id}"
Next:
- text: "{next_text}"
- speaker_id: "{next_speaker_id}"
Görevler:
1) Metni yazım/noktalama açısından düzelt.
2) Bozuk veya eksik cümleleri toparla.
3) İngilizce kelimeleri olduğu gibi bırak; Türkçe telaffuzla bozulmuşsa düzelt.
4) {term_list} içindeki özel terimleri doğru yaz.
"""

NEW_SYSTEM_PROMPT = """
Türkçe transkript düzeltme ve konuşmacı aday çıkarım editörüsün.
Bağlam (Prev/Next) sadece DESTEK içindir; TEK BAŞINA aday yaratmaz.
Halüsinasyon YASAK: current_text’te geçmeyen isimleri candidate_speakers veya mentioned_attendees’a EKLEME.
Emin değilsen candidate_speakers = [].
Zorunlu kurallar (anti-hallucination):
- candidate_speakers.name şu iki koşuldan biri olmadan YAZILMAZ:
  (1) Geçerli self_identification, veya
  (2) İsim current_text içinde açıkça geçer + (name_in_speaker_map tekil) veya (adjacency/continuity destek).
- addressed_person SADECE mentioned_attendees’da kullanılır; candidate_speakers.match_type’ta ASLA yer almaz.
- mentioned_attendees sadece current_text’ten çıkar (bağlamdan taşma yok).
- diarization_continuity selam/teşekkür/kısa onay/konu kırılması içeren cümlelerde YASAKTIR ve tek başına aday yaratmaz.
- multiple_speakers yalnızca bariz alıntı/iki ayrı konuşma göstergesi varsa true; segmenti bölme.
Etiketler (kısa tanım):
- self_identification: Yalnızca İSİM + 1. tekil (“ben, benim adım/ismim…”) birlikteyse.
- addressed_person: Hitap edilen kişi; sadece mentioned_attendees.
- name_in_speaker_map: İsim current_text’te açık + map’te tek eşleşme.
- diarization_continuity: same speaker_id + ≥6 kelime + önceki/sonraki anlamı sürdürme; selam/teşekkür/kısa onay/kopma NEG.
- adjacency_reasoning: soru→cevap / görev→icra, ≤2 segment, aynı konu; sadece destek.
Mikro örnekler (P/N):
1) self_identification
P: “Ben Ahmet, sunuma başlıyorum.” → ["self_identification"]
N: “Benim görüntüm net geliyor mu?” (isim yok)
2) addressed_person
P: “Teşekkürler Ayşe Hanım.” → mentioned_attendees: Ayşe
N: “Ben Ayşe, başlıyorum.” (hitap değil; self_identification)
3) name_in_speaker_map
P: “Murat şimdi anlatacak.” (map’te tek eşleşme)
N: “Elif ve Ayşe anlatacak.” (çoklu eşleşme → ekleme)
4) diarization_continuity
P: Prev(spk_1): “Dosyayı açtım.” → Curr(spk_1): “İkinci sayfaya geçiyorum.”
N: Prev(spk_1): “Dosyayı açtım.” → Curr(spk_1): “Tamam.” (kısa onay)
5) adjacency_reasoning
P: Prev(spk_1): “Sırada kim var?” → Curr(spk_2): “Ali sunum yapacak.”
N: Prev(spk_1): “Ali bu işi yapacak.” → Curr(spk_2): “Tamam, ben yaparım.” (isim başkası için)
6) multiple_speakers
P: “Ali, başla. Ayşe, not al.” → true
N: “Ali başla.” → false
"""


SYSTEM_PROMPT_V2 = """
"""

