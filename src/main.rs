use tokenizers::{Tokenizer, Result};

fn main() -> Result<()> {
    let input_str = "[국제발신] [해외결제] 확인코드:9**8 [KRW 959,000] 결제가 완료되었습니다. 배송관련 고객센터 070-7893-9043";

    let tokenizer = Tokenizer::from_pretrained("bert-base-multilingual-cased", None)?;
    let encoding = tokenizer.encode(input_str, false)?;
    let mut ids = encoding.get_ids().to_vec();
    ids = pad_sequences(ids, 128);
    let mut masks = encoding.get_attention_mask().to_vec();
    masks = pad_sequences(masks, 128);
    println!("{:?}", ids);
    println!("{:?}", masks);

    Ok(())
}

pub fn pad_sequences(mut input_ids: Vec<u32>, max_len: usize) -> Vec<u32> {
    let len = input_ids.len();
    if len < max_len {
        // If the sequence length is less than max_len, pad it with zeros.
        input_ids.resize(max_len, 0);
    } else if len > max_len {
        // If the sequence length is greater than max_len, truncate it.
        input_ids.truncate(max_len);
    }
    input_ids
}