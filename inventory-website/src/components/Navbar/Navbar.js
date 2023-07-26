const Navbar = () => {
    return (  
        <nav className="navbar">
            <h1>Invenlytics</h1>
            <div className="links">
                <a href="/">Dashboard</a>
                <a href="/inventory">Inventory List</a>
                <a href="/reports">Reports and Analytics</a>
                <a href="/account" style={{
                    color: "white",
                    backgroundColor: '#f47c32',
                    borderRadius: '8px'
                }}>Account</a>
            </div>
        </nav>
    );
}
 
export default Navbar;