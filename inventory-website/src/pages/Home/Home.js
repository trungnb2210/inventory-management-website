import { useState } from 'react';
import ProductList from '../../components/ProductList'

const Home = () => {
    const [products, setProducts] = useState([
        {name: '3x3 Rubik', sold: 900, stock: 250, producer: 'Rubik', id: 1},
        {name: 'Maze', sold: 600, stock: 40, producer: 'Smartgames', id: 2},
        {name: 'Sylvanian House', sold: 50, stock: 5, producer: 'Sylvanian', id: 3},
        {name: 'Hotwheels', sold: 1500, stock: 500, producer: 'Hotwheels', id: 4},
    ])

    return (  
        <div className="home">
            <ProductList products={products} title='All Products'></ProductList>
            <ProductList products={products.filter((product) => product.sold > 750)} title='Top Performing Products'></ProductList>
            <ProductList products={products.filter((product) => product.stock < 50)} title='Low Stock Products'></ProductList>
        </div>
    );
}
 
export default Home;
