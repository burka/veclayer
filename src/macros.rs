/// Macro to implement traits for Arc<T> where T implements the trait.
/// This eliminates boilerplate Arc delegation code across trait modules.
///
/// # Usage
///
/// For sync traits (methods returning concrete types):
/// ```ignore
/// arc_impl!(Embedder {
///     fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
///     fn dimension(&self) -> usize;
///     fn name(&self) -> &str;
/// });
/// ```
///
/// For async traits (methods returning impl Future):
/// ```ignore
/// arc_impl!(VectorStore {
///     fn insert_chunks(&self, chunks: Vec<HierarchicalChunk>) -> impl Future<Output = Result<()>> + Send;
///     fn search(&self, query_embedding: &[f32], limit: usize, level_filter: Option<ChunkLevel>) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
/// });
/// ```
#[macro_export]
macro_rules! arc_impl {
    // Main pattern: trait name followed by method signatures
    ($trait_name:ident {
        $(
            fn $method:ident(&self $(, $arg:ident: $arg_ty:ty)*) -> $ret:ty;
        )*
    }) => {
        impl<T: $trait_name + ?Sized> $trait_name for std::sync::Arc<T> {
            $(
                fn $method(&self $(, $arg: $arg_ty)*) -> $ret {
                    (**self).$method($($arg),*)
                }
            )*
        }
    };
}
