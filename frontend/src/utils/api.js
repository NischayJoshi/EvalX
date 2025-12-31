import axios from "axios";

const api = axios.create({
  baseURL: "https://viva-backend.yellowwater-5fddb32d.centralindia.azurecontainerapps.io/api",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;
//  ""
