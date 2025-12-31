import { Routes, Route } from 'react-router-dom'
import Hero from './Pages/Hero.jsx'
import Login from './Pages/Login.jsx'
import Signup from './Pages/Signup.jsx'
import OrganizerDashboard from './Pages/Organizer/OrganizerDashboard.jsx'
import EventDetails from './Pages/Organizer/EventDetails.jsx'
import DeveloperDashborad from './Pages/Developer/DeveloperDashborad.jsx'
import InterviewPage from './Pages/Developer/DevEventDetails/Tabs/InterviewPage.jsx'
// import DevEventDetails from './Pages/Developer/DevEventDetails.jsx'
import DevEventDetails from "./Pages/Developer/DevEventDetails/DevEventDetails.jsx"
import InterviewRoom from './Pages/Developer/DevEventDetails/Tabs/InterviewRoom.jsx'

function App() {
  return (
    <div className="app-container">
      <Routes>
        <Route path="/" element={<Hero />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* Organiser Routes here  */}
        <Route path="/org/dashboard/" element={ <OrganizerDashboard /> } />
        <Route path="/org/dashboard/id/:id" element={<EventDetails />} />

        {/* Developer Routes here  */}
        <Route path="/dashboard/" element={ <DeveloperDashborad /> } />
        <Route path="/dashboard/id/:id" element={<DevEventDetails />} />

        <Route path="/dashboard/interview" element={<InterviewPage />} />
        <Route path="/interview/:sessionId" element={<InterviewRoom />} />
      </Routes>
    </div>
  )
}

export default App
