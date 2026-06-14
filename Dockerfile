FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src

COPY ["VirtualFittingRoom/VirtualFittingRoom.csproj", "VirtualFittingRoom/"]
RUN dotnet restore "VirtualFittingRoom/VirtualFittingRoom.csproj"

COPY ["VirtualFittingRoom/", "VirtualFittingRoom/"]
WORKDIR /src/VirtualFittingRoom
RUN dotnet publish "VirtualFittingRoom.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS final
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgdiplus libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV ASPNETCORE_ENVIRONMENT=Production
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=false

COPY --from=build /app/publish .

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "ASPNETCORE_URLS=http://0.0.0.0:${PORT:-8080} dotnet VirtualFittingRoom.dll"]
