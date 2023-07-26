const ProductList = ({ products, title }) => {

    console.log(products, title)

    return (
        <div className="product-list">
            <h2> { title } </h2>
            {products.map((product) => (
                <div className="product-preview" key={product.id}>
                    <h2>{ product.name } by { product.producer }</h2>
                    <p>{ product.sold } sold with { product.stock } left </p>
                </div>
            ))}
        </div>
    )
}

export default ProductList;