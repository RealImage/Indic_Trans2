from indictrans import Transliterator
trn = Transliterator(source='<src_lang>', target='<trg_lang>', build_lookup=True)
tam_text = "<input_language>"
result = trn.transform(tam_text)
print(result)
