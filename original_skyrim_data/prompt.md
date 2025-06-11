Please write a script to convert my data manifest to the json format for training. 

# Desired Output Format for Training Dataset Manifest
The desired training dataset manifest is a list of jsonlines, meaning that each line is a JSON data in the following format:
```
Note:

    If timestamp training is not used, the sentences field can be excluded from the data.
    If data is only available for one language, the language field can be excluded from the data.
    If training empty speech data, the sentences field should be [], the sentence field should be "", and the language field can be absent.
    Data may exclude punctuation marks, but the fine-tuned model may lose the ability to add punctuation marks.

{
  "audio": {
    "path": "dataset/0.wav"
  },
  "sentence": "近几年，不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。",
  "language": "Chinese",
  "sentences": [
    {
      "start": 0,
      "end": 1.4,
      "text": "近几年，"
    },
    {
      "start": 1.42,
      "end": 8.4,
      "text": "不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。"
    }
  ],
  "duration": 7.37
}
```

Two files each: dataset/train.json and dataset/test.json

# Input: 
original_skyrim_data/test.yaml (test data manifest)
original_skyrim_data/train.yaml (train data manifest)

Format example for both files:
```yaml 
count: 11938
lines:
- DurationMs: 7198
  FileName: rigmorsigunnvoiceroc_rigmordlcq_rigmordlcquest0_000109b4_1.wav
  InternalFileName: rigmordlcq_rigmordlcquest0_000109b4_1.fuz
  Plugin: rigmorcyrodiil.esm
  Stage2Data: null
  State: ''
  Transcription: It's such a pity Ragnar isn't here to escort her to the ceremony.
    He would be so very proud of you both.
  VoiceType: rigmorsigunnvoiceroc
- DurationMs: 3947
  FileName: ztkvoicetypeisadore_ztkisadore02_idles_0006610b_1.wav
  InternalFileName: ztkisadore02_idles_0006610b_1.fuz
  Plugin: 0isadore.esp
  Stage2Data: null
  State: ''
  Transcription: I am okay with caves as long as there is no spiders or Trolls.
  VoiceType: ztkvoicetypeisadore
- DurationMs: 2414
  FileName: maleeventoned_dlc1vq01miscobjective__0000d8e9_1.wav
  InternalFileName: dlc1vq01miscobjective__0000d8e9_1.fuz
  Plugin: dawnguard.esm
  Stage2Data: null
  State: ''
  Transcription: Here to join the Dawnguard? Good.
  VoiceType: maleeventoned
- DurationMs: 4690
  FileName: malenord_mq301_mq301jarldragonsreac_000d23ca_1.wav
  InternalFileName: mq301_mq301jarldragonsreac_000d23ca_1.fuz
  Plugin: skyrim.esm
  Stage2Data: null
  State: ''
  Transcription: Ulfric and General Tullius are both just waiting for me to make a
    wrong move.
  VoiceType: malenord
- DurationMs: 11563
  FileName: femalenordvilja_aaemquestc_aaemsharedquest_0007b64a_1.wav
  InternalFileName: aaemquestc_aaemsharedquest_0007b64a_1.fuz
  Plugin: emcompviljaskyrim.esp
  Stage2Data: null
  State: ''
  Transcription: You know, I knew it right from the start... that Talen-Jei is in
    love with Keerava and wants to marry her. We really must help him find the amethysts
  VoiceType: femalenordvilja
- DurationMs: 5572
  FileName: malenord_dlc1hunterbaseintro__0001a3cb_1.wav
  InternalFileName: dlc1hunterbaseintro__0001a3cb_1.fuz
  Plugin: dawnguard.esm
  Stage2Data: null
  State: ''
  Transcription: If Isran hadn't left the Order, this could've been our home.
  VoiceType: malenord
- DurationMs: 10774
  FileName: maleeventonedaccented_dbrecurrin_dbrecurringcont_00087b7d_1.wav
  InternalFileName: dbrecurrin_dbrecurringcont_00087b7d_1.fuz
  Plugin: skyrim.esm
  Stage2Data: null
  State: ''
  Transcription: 'You must be the... assassin. I need you to kill the Itinerant Lumberjack,
    in Morthal, at the logging camp. Here, this is all the gold I have. '
  VoiceType: maleeventonedaccented
- DurationMs: 5340
  FileName: dwasptadisandavelvoice_dwaspsadrithkegranscene29__0094c4ab_1.wav
  InternalFileName: dwaspsadrithkegranscene29__0094c4ab_1.fuz
  Plugin: dwarfsphere.esp
  Stage2Data: null
  State: ''
  Transcription: We must have given them a discount then? The usual price is three
    thousand septims for a barrel.
  VoiceType: dwasptadisandavelvoice
- DurationMs: 8870
  FileName: inigofollowervoice_jrlucieninigo__00805dfa_1.wav
  InternalFileName: jrlucieninigo__00805dfa_1.fuz
  Plugin: lucien.esp
  Stage2Data: null
  State: ''
  Transcription: Oh, there once was a scholar named Lucien the Bright. He was slight
    but surprisingly good in a fight.
  VoiceType: inigofollowervoice
... (more lines)
```

The audio files are located in eg. original_skyrim_data/audio/000fcaltriusyoungvoicetype_000fcquest01__000a4f83_2.wav 