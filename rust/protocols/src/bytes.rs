use async_std::{
    io::{Read, Write},
    task,
};
use io_utils::imux::IMuxAsync;

#[inline]
pub async fn async_serialize<W, T>(w: &mut IMuxAsync<W>, value: &T) -> Result<(), bincode::Error>
where
    W: Write + Unpin,
    T: serde::Serialize + ?Sized,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    w.write(&bytes).await?;
    w.flush().await.map_err(|_| {
        Box::new(bincode::ErrorKind::Custom(
            "Error attempting to flush".to_string(),
        ))
    })
}

#[inline]
pub async fn async_deserialize<R, T>(reader: &mut IMuxAsync<R>) -> bincode::Result<T>
where
    R: Read + Unpin,
    T: serde::de::DeserializeOwned,
{
    let bytes = reader.read().await?;
    bincode::deserialize(&bytes[..])
}

#[inline]
pub fn serialize<W, T>(w: &mut IMuxAsync<W>, value: &T) -> Result<(), bincode::Error>
where
    W: Write + Unpin,
    T: serde::Serialize + ?Sized,
{
    task::block_on(async { async_serialize(w, value).await })
}

#[inline]
pub fn deserialize<R, T>(r: &mut IMuxAsync<R>) -> bincode::Result<T>
where
    R: Read + Unpin,
    T: serde::de::DeserializeOwned,
{
    task::block_on(async { async_deserialize(r).await })
}
