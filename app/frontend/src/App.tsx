import { createBrowserRouter, RouterProvider, NavLink, Navigate, Outlet } from 'react-router-dom'
import InputPage          from './pages/InputPage'
import RawPage            from './pages/RawPage'
import UnlabeledPage      from './pages/UnlabeledPage'
import LabelPage          from './pages/LabelPage'
import LabeledPage        from './pages/LabeledPage'
import InferencePage      from './pages/InferencePage'
import VideoAnnotatePage  from './pages/VideoAnnotatePage'

const nav = [
  { to: '/input',         label: 'Input' },
  { to: '/raw/train',     label: 'Train Raw' },
  { to: '/raw/val',       label: 'Val Raw' },
  { to: '/raw/test',      label: 'Test Raw' },
  { to: '/label',         label: 'Label' },
  { to: '/video-annotate', label: 'Video' },
  { to: '/labeled',       label: 'Labeled' },
  { to: '/infer',         label: 'Inference' },
]

function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-[#1a1a1a] text-gray-100 text-sm font-mono">
      <nav className="w-40 flex-shrink-0 bg-[#111] border-r border-[#2a2a2a] flex flex-col gap-0.5 p-2 pt-4">
        <p className="text-[10px] text-gray-600 uppercase tracking-widest px-2 mb-3">
          Salmon Annotate
        </p>
        {nav.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `px-3 py-2 rounded text-xs transition-colors ${
                isActive
                  ? 'bg-[#252525] text-white'
                  : 'text-gray-500 hover:text-gray-200 hover:bg-[#1e1e1e]'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="flex-1 overflow-hidden flex flex-col">
        <Outlet />
      </div>
    </div>
  )
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true,          element: <Navigate to="/input" replace /> },
      { path: 'input',        element: <InputPage /> },
      { path: 'raw/:split',   element: <RawPage /> },
      { path: 'label',           element: <LabelPage /> },
      { path: 'video-annotate',  element: <VideoAnnotatePage /> },
      { path: 'labeled',         element: <LabeledPage /> },
      { path: 'infer',           element: <InferencePage /> },
      { path: 'unlabeled',       element: <UnlabeledPage /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
