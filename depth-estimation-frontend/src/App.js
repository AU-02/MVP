import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import Footer from "./components/Footer";
import HomeScreen from "./screens/HomeScreen";
import GetStarted from "./screens/GetStarted";

const Layout = ({ children }) => {
  return (
    <>
      <NavBar />
      {children}
      <Footer />
    </>
  );
};

function App() {
  return (
    <Router>
      <Routes>
        {/* No Navbar/Footer */}
        <Route path="/" element={<GetStarted />} />

        {/* Protected Home Screen */}
        <Route path="/HomeScreen" element={<Layout><HomeScreen /></Layout>} />
      </Routes>
    </Router>
  );
}

export default App;
